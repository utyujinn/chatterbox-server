// The server process end to end over its pipes. The protocol part needs nothing; speaking needs the models and the GPU:
//   CB_ORT_DLL=<onnxruntime.dll>  cargo test --release --test server -- --ignored --nocapture
// with the folder of the CUDA/cuDNN DLLs on PATH.
use chatterbox_server::protocol::{read_frame, write_frame};
use serde_json::{json, Value};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Server {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Server {
    fn start() -> Server {
        let mut child = Command::new(env!("CARGO_BIN_EXE_chatterbox-server")).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let input = BufWriter::new(child.stdin.take().unwrap());
        let output = BufReader::new(child.stdout.take().unwrap());
        Server { child, input, output }
    }
    fn send(&mut self, header: Value, body: &[u8]) {
        write_frame(&mut self.input, &header, body).unwrap();
    }
    fn receive(&mut self) -> (Value, Vec<u8>) {
        let frame = read_frame(&mut self.output).unwrap().expect("the server closed its output");
        (frame.header, frame.body)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn requests_without_a_loaded_engine_are_answered_with_errors_and_the_process_lives_on() {
    let mut server = Server::start();
    for header in [json!({"cmd": "synthesize", "text": "Hi", "voice": "x"}), json!({"cmd": "register_voice", "out": "x"}), json!({"cmd": "nonsense"}), json!({})] {
        server.send(header, &[]);
        let (answer, _) = server.receive();
        assert_eq!(answer["ok"], false);
        assert!(answer["error"].as_str().is_some_and(|e| !e.is_empty()));
    }
    server.send(json!({"cmd": "load", "models_dir": "nowhere", "data_dir": root().join("data"), "ort_dll": "nowhere.dll"}), &[]);
    let (answer, _) = server.receive();
    assert_eq!(answer["ok"], false, "a load that cannot work is an error, not a crash");
}

#[test]
#[ignore]
fn the_server_loads_registers_a_voice_and_speaks() {
    let mut server = Server::start();
    let models = root().join("models");
    let voice_file = std::env::temp_dir().join(format!("chatterbox-server-test-{}.cbvoice", std::process::id()));
    server.send(
        json!({"cmd": "load", "models_dir": models, "data_dir": root().join("data"), "ort_dll": std::env::var("CB_ORT_DLL").unwrap(), "cuda": true}),
        &[],
    );
    assert_eq!(server.receive().0["ok"], true);
    let recording = std::fs::read(root().join("tests/golden/us_f_50s/ref.wav")).unwrap();
    server.send(json!({"cmd": "register_voice", "out": voice_file}), &recording);
    let (answer, _) = server.receive();
    assert_eq!(answer["ok"], true, "{answer}");

    // whole pieces
    server.send(json!({"cmd": "synthesize", "text": "Hello, nice to meet you. I just got back from the convention, and honestly I am tired.", "voice": voice_file, "seed": 1}), &[]);
    let mut pieces = 0;
    loop {
        let (header, body) = server.receive();
        assert_eq!(header["ok"], true, "{header}");
        if header["more"] == false {
            break;
        }
        assert_eq!(header["last"], true);
        assert!(body.len() > 10_000 && &body[..4] == b"RIFF");
        pieces += 1;
    }
    assert!(pieces >= 1);

    // streamed: several frames per piece, `last` only on the final one
    server.send(json!({"cmd": "synthesize", "text": "If you have a moment, I would love to show you around this world.", "voice": voice_file, "seed": 1, "stream": true}), &[]);
    let (mut frames, mut lasts) = (0, 0);
    loop {
        let (header, body) = server.receive();
        assert_eq!(header["ok"], true, "{header}");
        if header["more"] == false {
            break;
        }
        frames += 1;
        if header["last"] == true {
            lasts += 1;
        }
        assert!(&body[..4] == b"RIFF");
    }
    assert!(frames >= 2, "a sentence of this length comes as several chunks ({frames})");
    assert_eq!(lasts, 1);

    // errors do not stop the process
    server.send(json!({"cmd": "synthesize", "text": "Hi there.", "voice": "no-such-file.cbvoice"}), &[]);
    assert_eq!(server.receive().0["ok"], false);
    server.send(json!({"cmd": "synthesize", "text": "   ", "voice": voice_file}), &[]);
    assert_eq!(server.receive().0["ok"], false);
    server.send(json!({"cmd": "synthesize", "text": "Still alive?", "voice": voice_file, "temperature": 9.0}), &[]);
    assert_eq!(server.receive().0["ok"], false);
    server.send(json!({"cmd": "synthesize", "text": "Still alive and well.", "voice": voice_file, "seed": 2}), &[]);
    assert_eq!(server.receive().0["ok"], true);
    let _ = std::fs::remove_file(voice_file);
}

// The framing on the child's stdin / stdout, the same as irodori-server's (so the app's client code looks alike):
//   request  = u32 LE header length, header (JSON), u32 LE body length, body (bytes)
//   response = the same
// A request normally gets one response. `synthesize` gets one response per piece of text, each with `"more": true`, and a last
// one with `"more": false` and an empty body, so the app can start playing the first piece while the next is made.

use serde_json::Value;
use std::io::{self, Read, Write};

pub struct Frame {
    pub header: Value,
    pub body: Vec<u8>,
}

fn read_u32(input: &mut impl Read) -> io::Result<usize> {
    let mut bytes = [0u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes) as usize)
}

// A frame's parts are limited so that a damaged length cannot make the process ask for gigabytes.
const MAX_HEADER: usize = 1 << 20;
// A reference voice is a few megabytes; nothing the app sends comes near this.
const MAX_BODY: usize = 64 << 20;

// Ok(None) when the other end closed the pipe between two frames: that is how the app says "quit".
pub fn read_frame(input: &mut impl Read) -> io::Result<Option<Frame>> {
    let mut first = [0u8; 4];
    loop {
        match input.read(&mut first[..1]) {
            Ok(0) => return Ok(None),
            Ok(_) => break,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    input.read_exact(&mut first[1..])?;
    let header_len = u32::from_le_bytes(first) as usize;
    if header_len > MAX_HEADER {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "the header is too large"));
    }
    let mut header = vec![0u8; header_len];
    input.read_exact(&mut header)?;
    let body_len = read_u32(input)?;
    if body_len > MAX_BODY {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "the body is too large"));
    }
    let mut body = vec![0u8; body_len];
    input.read_exact(&mut body)?;
    // A header that is not JSON still ends where its length says, so the stream stays in step: it becomes a request without a
    // `cmd`, which the server answers with an error instead of ending the process.
    Ok(Some(Frame { header: serde_json::from_slice(&header).unwrap_or(Value::Null), body }))
}

pub fn write_frame(output: &mut impl Write, header: &Value, body: &[u8]) -> io::Result<()> {
    let header = serde_json::to_vec(header)?;
    output.write_all(&(header.len() as u32).to_le_bytes())?;
    output.write_all(&header)?;
    output.write_all(&(body.len() as u32).to_le_bytes())?;
    output.write_all(body)?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_frame_survives_the_round_trip() {
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &json!({"cmd": "x", "n": 3}), &[1, 2, 3]).unwrap();
        write_frame(&mut bytes, &json!({"cmd": "y"}), &[]).unwrap();
        let mut input = io::Cursor::new(bytes);
        let first = read_frame(&mut input).unwrap().unwrap();
        assert_eq!(first.header["n"], 3);
        assert_eq!(first.body, vec![1, 2, 3]);
        assert_eq!(read_frame(&mut input).unwrap().unwrap().header["cmd"], "y");
        assert!(read_frame(&mut input).unwrap().is_none(), "the end of the input is not an error");
    }

    #[test]
    fn a_header_that_is_not_json_does_not_break_the_stream() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(b"???");
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // an empty header
        bytes.extend_from_slice(&0u32.to_le_bytes());
        write_frame(&mut bytes, &json!({"cmd": "after"}), &[]).unwrap();
        let mut input = io::Cursor::new(bytes);
        assert!(read_frame(&mut input).unwrap().unwrap().header.is_null());
        assert!(read_frame(&mut input).unwrap().unwrap().header.is_null());
        assert_eq!(read_frame(&mut input).unwrap().unwrap().header["cmd"], "after");
    }

    #[test]
    fn a_frame_cut_short_is_an_error_and_a_huge_length_is_refused() {
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &json!({"cmd": "x"}), &[1, 2, 3, 4]).unwrap();
        bytes.truncate(bytes.len() - 2);
        assert!(read_frame(&mut io::Cursor::new(bytes)).is_err());
        let huge = (u32::MAX).to_le_bytes().to_vec();
        assert!(read_frame(&mut io::Cursor::new(huge)).is_err());
    }
}

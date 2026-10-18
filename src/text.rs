// The text side: the normalization of the original (`punc_norm`) and the cutting of a long text into pieces that are spoken one after
// the other (a piece is one sentence or so: the first sound of a text does not wait for the whole of it).

// What the original does to a text before the tokenizer (spec §4.1).
pub fn punc_norm(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }
    let mut chars = text.chars();
    let first = chars.next().unwrap();
    let mut text = if first.is_lowercase() { first.to_uppercase().collect::<String>() + chars.as_str() } else { text.to_string() };
    text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    for (from, to) in [("…", ", "), (":", ","), ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'")] {
        text = text.replace(from, to);
    }
    let mut text = text.trim_end_matches(' ').to_string();
    if !text.ends_with(['.', '!', '?', '-', ',']) {
        text.push('.');
    }
    text
}

const SHORTEST_PIECE: usize = 24;
const LONGEST_PIECE: usize = 300;

// Cuts after . ! ? (and the ellipsis) that end a sentence: followed by a space or the end, and not inside a number or a tag like
// "[laugh]". A piece shorter than SHORTEST_PIECE characters is joined to the next one (a lone "Hi." or "[laugh]" would be a sound of its
// own with the cost of a whole synthesis); a piece longer than LONGEST_PIECE is cut at the last comma or space before that.
pub fn split_pieces(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_tag = false;
    for (i, &c) in chars.iter().enumerate() {
        current.push(c);
        match c {
            '[' => in_tag = true,
            ']' => in_tag = false,
            '.' | '!' | '?' | '…' if !in_tag => {
                let next = chars.get(i + 1);
                let ends = next.is_none_or(|n| n.is_whitespace());
                if ends {
                    sentences.push(std::mem::take(&mut current));
                }
            }
            _ => {}
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current);
    }
    let mut pieces: Vec<String> = Vec::new();
    let mut carry = String::new();
    for sentence in sentences {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }
        let joined = if carry.is_empty() { sentence.to_string() } else { format!("{carry} {sentence}") };
        if joined.chars().count() < SHORTEST_PIECE {
            carry = joined;
        } else {
            carry.clear();
            pieces.extend(cut_long(&joined));
        }
    }
    if !carry.is_empty() {
        // a short rest: with the piece before it if there is one that is not too long, else on its own
        match pieces.last_mut() {
            Some(last) if last.chars().count() + carry.chars().count() < LONGEST_PIECE => {
                last.push(' ');
                last.push_str(&carry);
            }
            _ => pieces.push(carry),
        }
    }
    pieces
}

fn cut_long(sentence: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = sentence.trim().to_string();
    while rest.chars().count() > LONGEST_PIECE {
        let limit: usize = rest.char_indices().nth(LONGEST_PIECE).map(|(i, _)| i).unwrap_or(rest.len());
        let head = &rest[..limit];
        let cut = head.rfind(", ").map(|i| i + 1).or_else(|| head.rfind(' ')).unwrap_or(limit);
        out.push(rest[..cut].trim().to_string());
        rest = rest[cut..].trim_start().to_string();
    }
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punc_norm_does_what_the_original_does() {
        assert_eq!(punc_norm(""), "You need to add some text for me to talk.");
        assert_eq!(punc_norm("hello   world"), "Hello world.");
        // the original does not collapse the space after the replaced ellipsis either
        assert_eq!(punc_norm("Wait… really: yes"), "Wait,  really, yes.");
        assert_eq!(punc_norm("it’s “fine”!"), "It's \"fine\"!");
        assert_eq!(punc_norm("a , b -"), "A, b -");
        assert_eq!(punc_norm("Oh my gosh, [laugh] thank you"), "Oh my gosh, [laugh] thank you.");
    }

    #[test]
    fn sentences_become_pieces_and_short_ones_join() {
        let p = split_pieces("Hello, nice to meet you.");
        assert_eq!(p, vec!["Hello, nice to meet you."]);
        let p = split_pieces("Hi. I just got back from the convention, and honestly I am tired. Are you coming tomorrow? Yes!");
        assert_eq!(p, vec!["Hi. I just got back from the convention, and honestly I am tired.", "Are you coming tomorrow? Yes!"]);
        let p = split_pieces("The price is 3.5 dollars today, which is a lot. [laugh] Anyway.");
        assert_eq!(p, vec!["The price is 3.5 dollars today, which is a lot. [laugh] Anyway."]);
    }

    #[test]
    fn a_very_long_sentence_is_cut_and_nothing_is_lost() {
        let long = "word ".repeat(200);
        let pieces = split_pieces(&long);
        assert!(pieces.len() >= 3);
        assert!(pieces.iter().all(|p| p.chars().count() <= LONGEST_PIECE));
        assert_eq!(pieces.join(" ").split_whitespace().count(), 200);
    }

    #[test]
    fn no_text_gives_no_pieces() {
        assert!(split_pieces("   ").is_empty());
    }
}

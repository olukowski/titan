use std::collections::HashSet;

use super::{Document, Member, Number, Position, Span, TsfError, TsfResult, Value, ValueKind};

pub fn parse(file: Option<&str>, source: &str) -> TsfResult<Document> {
    Parser::new(file, source).parse_document()
}

struct Parser<'a> {
    file: Option<&'a str>,
    source: &'a str,
    offset: usize,
    position: Position,
}

impl<'a> Parser<'a> {
    fn new(file: Option<&'a str>, source: &'a str) -> Self {
        Self {
            file,
            source,
            offset: 0,
            position: Position { line: 1, column: 1 },
        }
    }

    fn parse_document(mut self) -> TsfResult<Document> {
        let comments = self.skip_ws_comments()?;
        let mut root = self.parse_value_with_comments(comments)?;
        let trailing = self.skip_ws_comments()?;
        if !trailing.is_empty() {
            root.comments.extend(trailing);
        }
        if !self.eof() {
            return Err(self.error(
                "TSF_PARSE_ERROR",
                "unexpected token after document",
                "",
                self.empty_span(),
            ));
        }
        Ok(Document {
            file: self.file.map(str::to_owned),
            root,
        })
    }

    fn parse_value_with_comments(&mut self, comments: Vec<String>) -> TsfResult<Value> {
        let start = self.position;
        let mut value = match self.peek_char() {
            Some('{') => self.parse_object()?,
            Some('[') => self.parse_array()?,
            Some('"') | Some('\'') => self.parse_string_value()?,
            Some('t') => self.parse_literal("true", ValueKind::Bool(true))?,
            Some('f') => self.parse_literal("false", ValueKind::Bool(false))?,
            Some('n') => {
                if self.starts_with("null") {
                    self.parse_literal("null", ValueKind::Null)?
                } else if self.starts_with("NaN") {
                    return Err(self.error_at(
                        "TSF_INVALID_NUMBER",
                        "NaN is not a valid TSF number",
                        "",
                        start,
                    ));
                } else {
                    return Err(self.error_at("TSF_PARSE_ERROR", "expected value", "", start));
                }
            }
            Some('N') if self.starts_with("NaN") => {
                return Err(self.error_at(
                    "TSF_INVALID_NUMBER",
                    "NaN is not a valid TSF number",
                    "",
                    start,
                ));
            }
            Some('I') if self.starts_with("Infinity") => {
                return Err(self.error_at(
                    "TSF_INVALID_NUMBER",
                    "Infinity is not a valid TSF number",
                    "",
                    start,
                ));
            }
            Some('+') => {
                return Err(self.error_at(
                    "TSF_INVALID_NUMBER",
                    "leading plus numbers are not valid TSF numbers",
                    "",
                    start,
                ));
            }
            Some('-') | Some('0'..='9') => self.parse_number_value()?,
            _ => return Err(self.error_at("TSF_PARSE_ERROR", "expected value", "", start)),
        };
        value.comments = comments;
        Ok(value)
    }

    fn parse_object(&mut self) -> TsfResult<Value> {
        let start = self.span_before();
        self.bump_char();
        let mut members = Vec::new();
        let mut keys = HashSet::new();
        loop {
            let comments = self.skip_ws_comments()?;
            if self.consume_char('}') {
                return Ok(Value {
                    kind: ValueKind::Object(members),
                    span: Span::merge(start, self.span_before()),
                    comments: Vec::new(),
                });
            }

            let (key, key_span) = self.parse_key()?;
            if !keys.insert(key.clone()) {
                return Err(self.error(
                    "TSF_DUPLICATE_KEY",
                    format!("duplicate object key '{key}'"),
                    "",
                    key_span,
                ));
            }
            self.skip_ws_comments()?;
            if !self.consume_char(':') {
                return Err(self.error(
                    "TSF_PARSE_ERROR",
                    "expected ':' after object key",
                    "",
                    self.empty_span(),
                ));
            }
            let value_comments = self.skip_ws_comments()?;
            let value = self.parse_value_with_comments(value_comments)?;
            members.push(Member {
                key,
                key_span,
                value,
                comments,
            });
            self.skip_ws_comments()?;
            if self.consume_char(',') {
                continue;
            }
            if self.consume_char('}') {
                return Ok(Value {
                    kind: ValueKind::Object(members),
                    span: Span::merge(start, self.span_before()),
                    comments: Vec::new(),
                });
            }
            return Err(self.error(
                "TSF_PARSE_ERROR",
                "expected ',' or '}'",
                "",
                self.empty_span(),
            ));
        }
    }

    fn parse_array(&mut self) -> TsfResult<Value> {
        let start = self.span_before();
        self.bump_char();
        let mut values = Vec::new();
        loop {
            let comments = self.skip_ws_comments()?;
            if self.consume_char(']') {
                return Ok(Value {
                    kind: ValueKind::Array(values),
                    span: Span::merge(start, self.span_before()),
                    comments: Vec::new(),
                });
            }
            values.push(self.parse_value_with_comments(comments)?);
            self.skip_ws_comments()?;
            if self.consume_char(',') {
                continue;
            }
            if self.consume_char(']') {
                return Ok(Value {
                    kind: ValueKind::Array(values),
                    span: Span::merge(start, self.span_before()),
                    comments: Vec::new(),
                });
            }
            return Err(self.error(
                "TSF_PARSE_ERROR",
                "expected ',' or ']'",
                "",
                self.empty_span(),
            ));
        }
    }

    fn parse_key(&mut self) -> TsfResult<(String, Span)> {
        match self.peek_char() {
            Some('"') | Some('\'') => self.parse_string(),
            Some(ch) if is_ident_start(ch) => self.parse_identifier(),
            _ => Err(self.error(
                "TSF_PARSE_ERROR",
                "expected object key",
                "",
                self.empty_span(),
            )),
        }
    }

    fn parse_string_value(&mut self) -> TsfResult<Value> {
        let (value, span) = self.parse_string()?;
        Ok(Value {
            kind: ValueKind::String(value),
            span,
            comments: Vec::new(),
        })
    }

    fn parse_string(&mut self) -> TsfResult<(String, Span)> {
        let start = self.span_before();
        let quote = self.bump_char().expect("string starts with quote");
        let mut out = String::new();
        while let Some(ch) = self.bump_char() {
            if ch == quote {
                return Ok((out, Span::merge(start, self.span_before())));
            }
            if ch == '\\' {
                let escaped = self.bump_char().ok_or_else(|| {
                    self.error(
                        "TSF_PARSE_ERROR",
                        "unterminated string escape",
                        "",
                        self.empty_span(),
                    )
                })?;
                match escaped {
                    '"' => out.push('"'),
                    '\'' => out.push('\''),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'v' => out.push('\u{000b}'),
                    '0' if !matches!(self.peek_char(), Some('0'..='9')) => out.push('\0'),
                    'x' => out.push(self.parse_hex_escape()?),
                    'u' => out.push(self.parse_unicode_escape()?),
                    '\n' => {}
                    '\r' => {
                        self.consume_char('\n');
                    }
                    _ => {
                        return Err(self.error(
                            "TSF_PARSE_ERROR",
                            "invalid string escape",
                            "",
                            self.empty_span(),
                        ));
                    }
                }
            } else {
                if ch == '\n' || ch == '\r' {
                    return Err(self.error(
                        "TSF_PARSE_ERROR",
                        "unterminated string",
                        "",
                        self.empty_span(),
                    ));
                }
                out.push(ch);
            }
        }
        Err(self.error(
            "TSF_PARSE_ERROR",
            "unterminated string",
            "",
            self.empty_span(),
        ))
    }

    fn parse_hex_escape(&mut self) -> TsfResult<char> {
        let mut value = 0_u32;
        for _ in 0..2 {
            let ch = self.bump_char().ok_or_else(|| {
                self.error(
                    "TSF_PARSE_ERROR",
                    "unterminated hex escape",
                    "",
                    self.empty_span(),
                )
            })?;
            value = value * 16
                + ch.to_digit(16).ok_or_else(|| {
                    self.error(
                        "TSF_PARSE_ERROR",
                        "invalid hex escape",
                        "",
                        self.empty_span(),
                    )
                })?;
        }
        char::from_u32(value).ok_or_else(|| {
            self.error(
                "TSF_PARSE_ERROR",
                "invalid hex scalar",
                "",
                self.empty_span(),
            )
        })
    }

    fn parse_unicode_escape(&mut self) -> TsfResult<char> {
        let mut value = 0_u32;
        for _ in 0..4 {
            let ch = self.bump_char().ok_or_else(|| {
                self.error(
                    "TSF_PARSE_ERROR",
                    "unterminated unicode escape",
                    "",
                    self.empty_span(),
                )
            })?;
            value = value * 16
                + ch.to_digit(16).ok_or_else(|| {
                    self.error(
                        "TSF_PARSE_ERROR",
                        "invalid unicode escape",
                        "",
                        self.empty_span(),
                    )
                })?;
        }
        char::from_u32(value).ok_or_else(|| {
            self.error(
                "TSF_PARSE_ERROR",
                "invalid unicode scalar",
                "",
                self.empty_span(),
            )
        })
    }

    fn parse_identifier(&mut self) -> TsfResult<(String, Span)> {
        let start = self.span_before();
        let mut ident = String::new();
        while let Some(ch) = self.peek_char() {
            if is_ident_continue(ch) {
                ident.push(ch);
                self.bump_char();
            } else {
                break;
            }
        }
        Ok((ident, Span::merge(start, self.span_before())))
    }

    fn parse_literal(&mut self, literal: &str, kind: ValueKind) -> TsfResult<Value> {
        let start = self.span_before();
        for expected in literal.chars() {
            if self.bump_char() != Some(expected) {
                return Err(self.error(
                    "TSF_PARSE_ERROR",
                    "invalid literal",
                    "",
                    self.empty_span(),
                ));
            }
        }
        Ok(Value {
            kind,
            span: Span::merge(start, self.span_before()),
            comments: Vec::new(),
        })
    }

    fn parse_number_value(&mut self) -> TsfResult<Value> {
        let start_offset = self.offset;
        let start = self.span_before();
        if self.consume_char('-') && self.peek_char().is_none() {
            return Err(self.error_at("TSF_INVALID_NUMBER", "invalid number", "", start.start));
        }
        if self.starts_with("0x") || self.starts_with("0X") {
            return Err(self.error_at(
                "TSF_INVALID_NUMBER",
                "hexadecimal numbers are not valid TSF numbers",
                "",
                start.start,
            ));
        }
        match self.peek_char() {
            Some('0') => {
                self.bump_char();
                if matches!(self.peek_char(), Some('0'..='9')) {
                    return Err(self.error_at(
                        "TSF_INVALID_NUMBER",
                        "numbers may not contain leading zeroes",
                        "",
                        start.start,
                    ));
                }
            }
            Some('1'..='9') => {
                while matches!(self.peek_char(), Some('0'..='9')) {
                    self.bump_char();
                }
            }
            _ => {
                return Err(self.error_at("TSF_INVALID_NUMBER", "invalid number", "", start.start));
            }
        }
        let mut had_fraction = false;
        if self.consume_char('.') {
            had_fraction = true;
            if !matches!(self.peek_char(), Some('0'..='9')) {
                return Err(self.error_at(
                    "TSF_INVALID_NUMBER",
                    "numbers may not have a trailing decimal point",
                    "",
                    start.start,
                ));
            }
            while matches!(self.peek_char(), Some('0'..='9')) {
                self.bump_char();
            }
        }
        if matches!(self.peek_char(), Some('e') | Some('E')) {
            self.bump_char();
            if matches!(self.peek_char(), Some('+') | Some('-')) {
                self.bump_char();
            }
            if !matches!(self.peek_char(), Some('0'..='9')) {
                return Err(self.error_at(
                    "TSF_INVALID_NUMBER",
                    "invalid exponent",
                    "",
                    start.start,
                ));
            }
            while matches!(self.peek_char(), Some('0'..='9')) {
                self.bump_char();
            }
        }
        let raw = &self.source[start_offset..self.offset];
        let value = raw.parse::<f64>().map_err(|_| {
            self.error(
                "TSF_INVALID_NUMBER",
                "invalid number",
                "",
                Span::merge(start, self.span_before()),
            )
        })?;
        if !value.is_finite() {
            return Err(self.error(
                "TSF_INVALID_NUMBER",
                "number must be finite",
                "",
                Span::merge(start, self.span_before()),
            ));
        }
        Ok(Value {
            kind: ValueKind::Number(Number {
                value,
                had_fraction,
            }),
            span: Span::merge(start, self.span_before()),
            comments: Vec::new(),
        })
    }

    fn skip_ws_comments(&mut self) -> TsfResult<Vec<String>> {
        let mut comments = Vec::new();
        loop {
            while matches!(self.peek_char(), Some(' ' | '\t' | '\r' | '\n')) {
                self.bump_char();
            }
            if self.starts_with("//") {
                self.bump_char();
                self.bump_char();
                let mut text = String::from("//");
                while let Some(ch) = self.peek_char() {
                    if ch == '\n' || ch == '\r' {
                        break;
                    }
                    text.push(ch);
                    self.bump_char();
                }
                comments.push(text);
            } else if self.starts_with("/*") {
                self.bump_char();
                self.bump_char();
                let mut text = String::from("/*");
                loop {
                    let ch = self.bump_char().ok_or_else(|| {
                        self.error(
                            "TSF_PARSE_ERROR",
                            "unterminated block comment",
                            "",
                            self.empty_span(),
                        )
                    })?;
                    text.push(ch);
                    if ch == '*' && self.consume_char('/') {
                        text.push('/');
                        break;
                    }
                }
                comments.push(text);
            } else {
                return Ok(comments);
            }
        }
    }

    fn starts_with(&self, text: &str) -> bool {
        self.source[self.offset..].starts_with(text)
    }

    fn consume_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.bump_char();
            true
        } else {
            false
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.source[self.offset..].chars().next()
    }

    fn bump_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.offset += ch.len_utf8();
        if ch == '\n' {
            self.position.line += 1;
            self.position.column = 1;
        } else {
            self.position.column += 1;
        }
        Some(ch)
    }

    fn eof(&self) -> bool {
        self.offset >= self.source.len()
    }

    fn span_before(&self) -> Span {
        Span {
            start: self.position,
            end: self.position,
        }
    }

    fn empty_span(&self) -> Span {
        Span {
            start: self.position,
            end: self.position,
        }
    }

    fn error(
        &self,
        code: &str,
        message: impl Into<String>,
        path: impl Into<String>,
        span: Span,
    ) -> TsfError {
        TsfError::one(self.file, code, message, path, span)
    }

    fn error_at(
        &self,
        code: &str,
        message: impl Into<String>,
        path: impl Into<String>,
        position: Position,
    ) -> TsfError {
        self.error(
            code,
            message,
            path,
            Span {
                start: position,
                end: position,
            },
        )
    }
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch == '$' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    is_ident_start(ch) || ch.is_ascii_digit() || ch == '-'
}

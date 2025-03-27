use log::LevelFilter;

use std::env;

pub type ErrBox = Box<dyn std::error::Error>;

/// Init logging at info level by default
pub fn init_log() {
    match env::var("RUST_LOG") {
        Ok(_value) => env_logger::init(),
        Err(_e) => env_logger::Builder::new()
            .filter_level(LevelFilter::Info)
            .init(),
    }
}

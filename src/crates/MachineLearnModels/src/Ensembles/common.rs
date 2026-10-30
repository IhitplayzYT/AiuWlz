#[derive(Debug,Clone)]
pub enum Voting{
    Hard,
    Soft
}

#[derive(Debug,Clone)]
pub enum Sampling{
    Bagging,
    Pasting,
}

pub struct Configured;
pub struct Trained;
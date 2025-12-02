use crossbeam_channel::Sender;

pub struct ShardedSender<T: Send + 'static> {
    shards: Vec<Sender<T>>,
    len: usize,
    counter: usize,
}

pub fn new_sharded_sender<T: Send + 'static>(shards: Vec<Sender<T>>) -> ShardedSender<T> {
    let len = shards.len();
    assert!(len > 0, "shards cannot be empty");
    ShardedSender {
        shards,
        len,
        counter: 0,
    }
}

impl<T: Send + 'static> ShardedSender<T> {
    pub fn send(&mut self, item: T) -> Result<(), crossbeam_channel::SendError<T>> {
        let idx = self.counter % self.len;
        self.counter = self.counter.wrapping_add(1);

        self.shards[idx].send(item)
    }

    pub fn try_send(&mut self, item: T) -> Result<(), crossbeam_channel::TrySendError<T>> {
        let idx = self.counter % self.len;
        self.counter = self.counter.wrapping_add(1);

        self.shards[idx].try_send(item)
    }
}

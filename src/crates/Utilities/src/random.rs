pub mod Random{
// Xoshiro256** PRNG

    use std::f64::consts::PI;

    pub struct Rng {
        s: [u64; 4],
    }

    impl Rng {
        // Spreading seed over 4 xoshiro states 
        pub fn seed(seed: u64) -> Self {
            let mut z = seed;
            let mut next = || {
                z = z.wrapping_add(0x9E3779B97F4A7C15);
                let mut x = z;
                x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
                x ^ (x >> 31)
            };
            Rng { s: [next(), next(), next(), next()] }
        }

        pub fn next_u64(&mut self) -> u64 {
            let r = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);
            let t = self.s[1] << 17;
            self.s[2] ^= self.s[0];
            self.s[3] ^= self.s[1];
            self.s[1] ^= self.s[2];
            self.s[0] ^= self.s[3];
            self.s[2] ^= t;
            self.s[3] = self.s[3].rotate_left(45);
            r
        }

        // float ISO trick for uniform disrib btwn [0,1)
        pub fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        // normal distrib using box Muller
        pub fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(f64::MIN_POSITIVE); // Avoid u1 becoming 0 for ln
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
    }
}
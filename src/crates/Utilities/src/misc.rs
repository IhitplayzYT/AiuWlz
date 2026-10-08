pub trait IntLike{}
pub trait FloatLike{}
pub trait SignedIntLike{}

impl IntLike for i8{}
impl IntLike for i16{}
impl IntLike for i32{}
impl IntLike for i64{}
impl IntLike for isize{}
impl IntLike for u8{}
impl IntLike for u16{}
impl IntLike for u32{}
impl IntLike for u64{}
impl IntLike for usize{}

impl FloatLike for f32{}
impl FloatLike for f64{}

impl SignedIntLike for i8{}
impl SignedIntLike for i16{}
impl SignedIntLike for i32{}
impl SignedIntLike for i64{}
impl SignedIntLike for isize{}
impl SignedIntLike for f32{}
impl SignedIntLike for f64{}

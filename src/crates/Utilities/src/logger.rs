use std::{collections::HashMap, error::Error, fs, hash::Hash, os::fd::RawFd, path::{Path, PathBuf}};

use chrono::Utc;

#[allow(non_snake_case,non_camel_case_types,dead_code,unreachable_code,unused)]
pub struct Logger{
    pub fflush: usize,
    pub curf: usize,
    pub sflush: usize,
    pub buff: Vec<u8>,
    pub metadata: HashMap<String,String>,
    pub output_stream: Option<OutputStream>,
    pub fmt: Option<String>
}

pub enum t_Logger{
    Frequency(usize),
    Size(usize),
    SizedFrequent(usize,usize)
}


pub enum LogType{
    Warn,
    Info,
    Error,
    Success,
    Debug,
    Trace,
    Log
}


pub enum OutputStream{
    File(String),
    Window(usize),
    FD(u16)
}


impl Logger{

    pub fn new(ty: t_Logger,op: Option<OutputStream>,meta:Option<HashMap<String,String>>) -> Self{
        let mut ret = Self{curf:0,fmt:None,fflush:0,sflush:0,buff:vec![],metadata:HashMap::new(),output_stream:None};
        match ty{
            t_Logger::Frequency(f) => {ret.fflush = f;}
            t_Logger::Size(s) => {ret.sflush = s;}
            t_Logger::SizedFrequent(s,f) => {ret.fflush = f;ret.sflush = s;}
        }
        ret.output_stream = op;
        if let Some(meta) = meta{
            ret.metadata = meta;
        }
        ret
    }

    fn flush(&mut self) {
    }

    pub fn LOG(&mut self,log_lvl:LogType,mssg: &str) -> Result<(),Box<dyn Error>>{
        let logstr = if let Some(fmt) = &self.fmt{
            fmt.replace("{LOG_CATEGORY}",stringify!(log_lvl)).replace("{LOG_LEVEL}", stringify!(log_lvl)).replace("{MSSG}", mssg).replace("{TIMESTAMP}",&Utc::now().to_string())
        }else{
            format!("[{}]{}: {}\n",stringify!(log_lvl),Utc::now().to_string(),mssg)
        };
        match (self.fflush,self.sflush) {
            (0,x) => {
              if self.buff.len() + logstr.len() <= x{
                self.buff.append(&mut logstr[..].as_bytes().to_vec());
              }else{
                if let Some(z) = &self.output_stream{
                        match z{
                            OutputStream::Window(id) => {
                                todo!();
                                //let window_handle = GetWindow(id);
                                //window_handle.write(logstr);

                            },
                            OutputStream::FD(fd) => {
                                let fd = RawFd::from(*fd);
                                let errcode = unsafe{ libc::write(fd, self.buff.as_ptr().cast(),self.buff.len())};
                                if errcode <= 0{return Err(Box::new(std::io::Error::from(std::io::ErrorKind::Other)))}
                                self.buff.append(&mut logstr.as_bytes().to_vec());
                            },
                            OutputStream::File(pth)  => {
                                let buff = self.buff.clone();
                                self.buff.clear();
                                fs::write(&pth,fs::read_to_string(pth)? + &String::from_utf8(buff)?)?;
                                self.buff.append(&mut logstr.as_bytes().to_vec());
                            }
                        }
                    }
                    else{
                        let buff = self.buff.clone();
                        self.buff.clear();
                        println!("{}",String::from_utf8(buff)?);
                        self.buff.append(&mut logstr.as_bytes().to_vec());
                    }
              }
            },
            (x,0) => {
            if self.curf <= x{
                self.buff.append(&mut logstr[..].as_bytes().to_vec());
                self.curf += 1;
              }else{
                if let Some(z) = &self.output_stream{
                        match z{
                            OutputStream::Window(id) => {
                                todo!();
                                //let window_handle = GetWindow(id);
                                //window_handle.write(logstr);
                                // self.curf = 1
                            },  
                            OutputStream::FD(fd) => {
                                let fd = RawFd::from(*fd);
                                let errcode = unsafe{ libc::write(fd, self.buff.as_ptr().cast(),self.buff.len())};
                                if errcode <= 0{return Err(Box::new(std::io::Error::from(std::io::ErrorKind::Other)))}
                                self.buff.append(&mut logstr.as_bytes().to_vec());
                                self.curf = 1;
                            },
                            OutputStream::File(pth)  => {
                                let buff = self.buff.clone();
                                self.buff.clear();
                                fs::write(&pth,fs::read_to_string(pth)? + &String::from_utf8(buff)?)?;
                                self.buff.append(&mut logstr.as_bytes().to_vec());
                                self.curf = 1;
                            }
                        }
                    }
                    else{
                        let buff = self.buff.clone();
                        self.buff.clear();
                        println!("{}",String::from_utf8(buff)?);
                        self.buff.append(&mut logstr.as_bytes().to_vec());
                        self.curf = 1;
                    }
              }

            }
            (x,y) => {
                // TODO: //FIXME:
                todo!();
            if self.curf <= x{
                self.buff.append(&mut logstr[..].as_bytes().to_vec());
                self.curf += 1;
              }else{
                if let Some(z) = &self.output_stream{
                        match z{
                            OutputStream::Window(id) => {
                                todo!();
                                //let window_handle = GetWindow(id);
                                //window_handle.write(logstr);
                                // self.curf = 1
                            },  
                            OutputStream::FD(fd) => {
                                let fd = RawFd::from(*fd);
                                let errcode = unsafe{ libc::write(fd, self.buff.as_ptr().cast(),self.buff.len())};
                                if errcode <= 0{return Err(Box::new(std::io::Error::from(std::io::ErrorKind::Other)))}
                                self.buff.append(&mut logstr.as_bytes().to_vec());
                                self.curf = 1;
                            },
                            OutputStream::File(pth)  => {
                                let buff = self.buff.clone();
                                self.buff.clear();
                                fs::write(&pth,fs::read_to_string(pth)? + &String::from_utf8(buff)?)?;
                                self.buff.append(&mut logstr.as_bytes().to_vec());
                                self.curf = 1;
                            }
                        }
                    }
                    else{
                        let buff = self.buff.clone();
                        self.buff.clear();
                        println!("{}",String::from_utf8(buff)?);
                        self.buff.append(&mut logstr.as_bytes().to_vec());
                        self.curf = 1;
                    }
              }

            }
        }

        Ok(())
    } 



}





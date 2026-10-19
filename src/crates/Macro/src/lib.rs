use proc_macro::TokenStream;
use proc_macro2::Span;
use syn::{Block, Ident, Pat, Stmt, parse_macro_input};
use Tensors::autograd;


#[proc_macro]
pub fn with_grad(input: TokenStream) -> TokenStream{
    let parsed = parse_macro_input!(input as Block);
    quote::quote! {
        autograd::with_grad( || #parsed)
    }.into()
}

// #[proc_macro]
// pub fn autograd(input: TokenStream) -> TokenStream{
//     let parsed = parse_macro_input!(input as Block);
//     let mut ret:TokenStream = quote::quote! {
//         autograd::with_grad( || #parsed)
//     }.into();
// 
//     let last_var = parsed.stmts.iter().rev().find_map(|x|{
//         if let Stmt::Local(last) = x{
//             if let Pat::Ident(ident) = &last.pat {
//                 return Some(ident.ident.to_string());
//             }
//         }
//         None
//     });
//     
//     if let Some(last_varib) = last_var{
//         let last_varib = Ident::new(&last_varib, Span::call_site());
//         ret.extend(quote::quote! {
//             #last_varib.backward();
//         });
//         ret
//     }else{
//         ret
//     }
// 
// }



use candle_core::{DType, Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let a = Tensor::zeros((3, 5), DType::U32, &device)?;

    println!("{a}, type:{:?}", a.dtype());

    println!("--- slice_assign ---");
    let values = Tensor::new(&[[1u32]], &device)?;
    let a = a.slice_assign(&[1..2, 3..4], &values)?;
    println!("{a}");

    println!(
        "value by index: {:?}",
        a.get(1)?.get(3)?.to_scalar::<u32>()?
    );
    println!("value by index: {:?}", a.to_vec2::<u32>()?[1][3]);

    println!("--- scatter_add, update row ---");
    let indices = Tensor::new(&[[2u32; 5]; 1], &device)?;
    let one = Tensor::new(&[[1u32; 5]], &device)?;
    let a = a.scatter_add(&indices, &one, 0 /*by row 0, by col 1*/)?;
    println!("{a}");

    println!("--- scatter_add, update column ---");
    let indices = Tensor::new(&[[2u32]; 3], &device)?;
    let one = Tensor::new(&[[1u32]; 3], &device)?;
    let a = a.scatter_add(&indices, &one, 1 /*by row 0, by col 1*/)?;
    println!("{a}");

    Ok(())
}

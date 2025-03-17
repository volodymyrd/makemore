use candle_core::backprop::GradStore;
use candle_core::Var;
use std::collections::{HashMap, HashSet};
use std::ops::Sub;
use std::{fs, io};
use std::time::Instant;

fn read_data() -> io::Result<Vec<String>> {
    fs::read_to_string("../../data/names.txt")
        .map(|contents| contents.lines().map(String::from).collect())
}

fn vocabulary(words: &Vec<String>) -> Vec<char> {
    let mut chars = words
        .iter()
        .flat_map(|word| word.chars())
        .collect::<HashSet<_>>()
        .drain()
        .collect::<Vec<_>>();
    chars.sort_unstable();
    chars
}

fn s_to_i(chars: &Vec<char>) -> HashMap<char, usize> {
    chars.iter().enumerate().map(|(i, &c)| (c, i + 1)).collect()
}

fn i_to_s(stoi: &HashMap<char, usize>) -> HashMap<usize, char> {
    stoi.iter().map(|(&c, &i)| (i, c)).collect()
}

fn training_set(words: &[String], stoi: &HashMap<char, usize>) -> (Vec<usize>, Vec<usize>) {
    let mut xs = vec![];
    let mut ys = vec![];

    for w in words {
        let mut chs = vec!['.'];
        chs.extend(w.chars());
        chs.push('.');

        for window in chs.windows(2) {
            if let &[ch1, ch2] = window {
                xs.push(stoi[&ch1]);
                ys.push(stoi[&ch2]);
            }
        }
    }

    (xs, ys)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = read_data()?;
    println!("{:?}", &words[..8]);
    println!("{}", words.len());

    let chars = vocabulary(&words);
    println!("vocabulary: {chars:?} and len is {}", chars.len());

    let mut stoi = s_to_i(&chars);
    stoi.insert('.', 0);
    println!("{stoi:?}");

    let itos = i_to_s(&stoi);
    println!("{itos:?}");

    println!("-----training sets-----");
    let (xs, ys) = training_set(&words[..1], &stoi);
    println!("{xs:?}");
    println!("{ys:?}");

    println!("-----one hot encoding-----");
    use candle_core::{Device, Tensor};
    use candle_nn::encoding::one_hot;

    let device = Device::Cpu;

    let xs: Vec<_> = xs.into_iter().map(|e| e as u32).collect();
    let xs = Tensor::new(xs, &device)?;

    // https://docs.rs/candle-nn/latest/candle_nn/encoding/fn.one_hot.html
    let xenc = one_hot(xs, 27, 1f32, 0f32)?;
    println!("{:?}", xenc.shape());
    println!("{:?}", xenc.to_vec2::<f32>()?);

    println!("-----define neuron-----");
    let W = Tensor::randn(0f32, 1f32, &[27, 27], &device)?;
    let X = xenc.matmul(&W)?;
    fn multiply(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut v = 0f32;
        for i in 0..a.len() {
            v += a[i] * b[i];
        }
        v
    }
    let X_3_13 = X.to_vec2::<f32>()?[3][13];
    let mm = multiply(
        &xenc.to_vec2::<f32>()?[3],
        &W.to_vec2::<f32>()?.iter().map(|r| r[13]).collect(),
    );
    assert_eq!(X_3_13, mm);
    println!("X[3, 13]={X_3_13} = (xenc[3] * W[:, 13]).sum()={mm}");

    println!("-----SUMMARY-----");
    let (xs, ys) = training_set(&words[..1], &stoi);
    let xs: Vec<_> = xs.into_iter().map(|e| e as u32).collect();
    let ys: Vec<_> = ys.into_iter().map(|e| e as u32).collect();

    let xs = Tensor::new(xs, &device)?;
    let ys = Tensor::new(ys, &device)?;

    let xenc = one_hot(xs.clone(), 27, 1f32, 0f32)?;
    let W = Tensor::randn(0f32, 1f32, &[27, 27], &device)?;
    let logits = xenc.matmul(&W)?;
    let counts = logits.exp()?;
    println!("counts shape: {:?}", counts.shape());
    let sum = counts.sum_keepdim(1)?;
    println!("sum shape: {:?}", sum.shape());
    let probs = counts.broadcast_div(&sum)?;
    println!("probs shape: {:?}", probs.shape());

    println!("---START EXPLANATION---");
    let mut nlls = [0.0; 5];
    let xs_vec: Vec<u32> = xs.to_vec1()?;
    let ys_vec: Vec<u32> = ys.to_vec1()?;
    let probs_vec: Vec<Vec<f32>> = probs.to_vec2()?;
    for i in 0..5 {
        // i-th bigram
        let x = xs_vec[i] as usize; // input character index
        let y = ys_vec[i] as usize; // label character index
        println!("----------");
        println!(
            "bigram example {}: {}{} (indexes {x},{y})",
            i + 1,
            itos[&x],
            itos[&y]
        );
        println!("input to the neural net: {x}[{}]", itos[&x]);
        println!("output probabilities from neural net: {:?}", probs_vec[i]);
        println!("label (actual next character): {y}[{}]", itos[&y]);
        let p = probs_vec[i][y];
        println!(
            "probability assigned by the next to the correct character: {}",
            p
        );
        let logp: Vec<Vec<f32>> = probs.log()?.to_vec2()?;
        let logp = logp[i][y];
        println!("log likelihood: {}", logp);
        let nll = -logp;
        println!("negative log likelihood: {}", nll);
        nlls[i] = nll;
    }
    let nlls = Tensor::new(&nlls, &device)?;

    println!("=========");
    let calc_loss = nlls.mean(0)?;
    println!("average negative log likelihood, i.e. loss= {calc_loss}");

    let vectorized_loss = Tensor::new(
        (0..5)
            .into_iter()
            .zip(ys_vec.iter())
            .map(|(row, &col)| probs_vec[row as usize][col as usize])
            .collect::<Vec<_>>(),
        &device,
    )?
    .log()?
    .mean(0)?
    .neg()?;
    println!("vectorized loss= {}", vectorized_loss);
    assert_eq!(
        calc_loss.to_vec0::<f32>()?,
        vectorized_loss.to_vec0::<f32>()?
    );
    println!("---END EXPLANATION---");

    let start = Instant::now();
    let (xs, ys) = training_set(&words, &stoi);
    let xs: Vec<_> = xs.into_iter().map(|e| e as u32).collect();
    let ys: Vec<_> = ys.into_iter().map(|e| e as u32).collect();
    let nums = xs.len();
    println!("number of examples:{nums}");

    let xs = Tensor::new(xs, &device)?;
    let ys = Var::from_tensor(&Tensor::new(ys, &device)?)?;

    let mut W = Var::from_tensor(&Tensor::randn(0f32, 1f32, &[27, 27], &device)?)?;
    let xenc = Var::from_tensor(&one_hot(xs, 27, 1f32, 0f32)?)?;

    for k in 0..100 {
        // forward pass
        let logits = xenc.matmul(&W)?;
        let counts = logits.exp()?;
        let sum = counts.sum_keepdim(1)?;
        let probs = counts.broadcast_div(&sum)?;

        let loss = probs
            .gather(&ys.unsqueeze(1)?, 1)?
            .squeeze(1)?
            .log()?
            .mean(0)?
            .neg()?;

        println!("iter:{k}, loss: {}", loss.to_vec0::<f32>()?);

        // backward pass
        let grad_store = loss.backward()?;
        let w_grad = grad_store
            .get(&W)
            .unwrap()
            .broadcast_mul(&Tensor::new(50f32, &device)?)?;
        let update = W.as_tensor().sub(&w_grad)?;
        W.set(&update)?;
    }
    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);

    Ok(())
}

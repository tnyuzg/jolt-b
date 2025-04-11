use ark_serialize::CanonicalSerialize;
use ark_std::test_rng;
use goldilocks::{Goldilocks, GoldilocksExt2};
use rand_core::RngCore;
use sha2::Sha256;
use std::time::Instant;

use crate::{
    poly::{
        commitment::{
            basefold::{BasefoldCommitmentScheme, BasefoldPP, BASEFOLD_ADDITIONAL_RATE_BITS},
            commitment_scheme::{BatchType, CommitmentScheme},
        },
        dense_mlpoly::DensePolynomial,
        field::JoltField,
    },
    utils::transcript::ProofTranscript,
};

fn test_basefold_helper(num_vars: usize, rng: &mut impl RngCore) {
    let pp = BasefoldPP::<Goldilocks, GoldilocksExt2, Sha256>::new(BASEFOLD_ADDITIONAL_RATE_BITS);

    let poly_evals = (0..(1 << num_vars))
        .map(|_| Goldilocks::random(rng))
        .collect();
    let poly = DensePolynomial::new(poly_evals);

    let opening_point: Vec<_> = (0..num_vars).map(|_| Goldilocks::random(rng)).collect();
    let eval = poly.evaluate(&opening_point);

    let now = Instant::now();
    let commitment = BasefoldCommitmentScheme::commit(&poly, &pp);
    println!("committing elapsed {}", now.elapsed().as_millis());

    let mut prover_transcript = ProofTranscript::new(b"example");
    let mut verifier_transcript = ProofTranscript::new(b"example");

    let now = Instant::now();
    let eval_proof = BasefoldCommitmentScheme::prove(
        &poly,
        &pp,
        &commitment,
        &opening_point,
        &mut prover_transcript,
    );
    println!("proving elapsed {}", now.elapsed().as_millis());

    let now = Instant::now();
    BasefoldCommitmentScheme::verify(
        &eval_proof,
        &pp,
        &mut verifier_transcript,
        &opening_point,
        &eval,
        &commitment,
    )
    .unwrap();
    println!("verifying elapsed {}", now.elapsed().as_millis());
}

#[test]
fn test_basefold_vanilla() {
    let mut rng = test_rng();

    for i in 5..=18 {
        for _ in 0..10 {
            test_basefold_helper(i, &mut rng);
        }
    }
}

fn test_basefold_batch_helper(num_vars: usize, batch_size: usize, rng: &mut impl RngCore) {
    let pp = BasefoldPP::<Goldilocks, GoldilocksExt2, Sha256>::new(BASEFOLD_ADDITIONAL_RATE_BITS);

    let poly_evals: Vec<Vec<_>> = (0..batch_size)
        .map(|_| {
            (0..(1 << num_vars))
                .map(|_| Goldilocks::random(rng))
                .collect()
        })
        .collect();

    let polys: Vec<_> = (0..batch_size)
        .map(|i| DensePolynomial::new(poly_evals[i].clone()))
        .collect();

    let opening_point: Vec<_> = (0..num_vars).map(|_| Goldilocks::random(rng)).collect();

    let evals: Vec<_> = polys
        .iter()
        .map(|poly| poly.evaluate(&opening_point))
        .collect();

    let now = Instant::now();
    let commitments = BasefoldCommitmentScheme::batch_commit_polys(&polys, &pp, BatchType::Big);
    println!("committing elapsed {}", now.elapsed().as_millis());

    let mut prover_transcript = ProofTranscript::new(b"example");
    let mut verifier_transcript = ProofTranscript::new(b"example");

    let now = Instant::now();
    let batch_proof = BasefoldCommitmentScheme::batch_prove(
        &polys.iter().collect::<Vec<_>>(),
        &pp,
        &commitments.iter().collect::<Vec<_>>(),
        &opening_point,
        &evals,
        BatchType::Big,
        &mut prover_transcript,
    );
    println!("proving elapsed {}", now.elapsed().as_millis());

    let now = Instant::now();
    BasefoldCommitmentScheme::batch_verify(
        &batch_proof,
        &pp,
        &opening_point,
        &evals,
        &commitments.iter().collect::<Vec<_>>(),
        &mut verifier_transcript,
    )
    .unwrap();
    println!("verifying elapsed {}", now.elapsed().as_millis());
}

#[test]
fn test_basefold_batch() {
    let mut rng = test_rng();

    for num_vars in 5..=18 {
        for batch_size in 1..=10 {
            test_basefold_batch_helper(num_vars, batch_size, &mut rng);
        }
    }
}


pub fn basefold_pack_batch_benchmark(
    output_path: &str,
    n: usize,
    k: usize,
) -> std::io::Result<()> {
    use std::time::Instant;
    let mut wtr = csv::Writer::from_path(output_path)?;
    wtr.write_record([
        "num_vars",
        "batch_size",
        "commit_ms",
        "prove_ms",
        "verify_ms",
        "proof_size_MB",
    ])?;

    let mut rng = test_rng();

    for num_vars in (n - k)..=n {
        let batch_size = 1 << (n - num_vars); // 2^(n - num_vars)

        let pp = BasefoldPP::<GoldilocksExt2, GoldilocksExt2, Sha256>::new(BASEFOLD_ADDITIONAL_RATE_BITS);

        // --- 构造 polys ---
        let polys: Vec<_> = (0..batch_size)
            .map(|_| {
                let evals = (0..(1 << num_vars))
                    .map(|_| GoldilocksExt2::random(&mut rng))
                    .collect();
                DensePolynomial::new(evals)
            })
            .collect();

        let opening_point: Vec<_> = (0..num_vars)
            .map(|_| GoldilocksExt2::random(&mut rng))
            .collect();
        let evals: Vec<_> = polys
            .iter()
            .map(|poly| poly.evaluate(&opening_point))
            .collect();

        // --- Commit ---
        let now = Instant::now();
        let commitments =
            BasefoldCommitmentScheme::batch_commit_polys(&polys, &pp, BatchType::Big);
        let commit_ms = now.elapsed().as_millis();

        // --- Prove ---
        let mut prover_transcript = ProofTranscript::new(b"example");
        let now = Instant::now();
        let proof = BasefoldCommitmentScheme::batch_prove(
            &polys.iter().collect::<Vec<_>>(),
            &pp,
            &commitments.iter().collect::<Vec<_>>(),
            &opening_point,
            &evals,
            BatchType::Big,
            &mut prover_transcript,
        );
        let prove_ms = now.elapsed().as_millis();

        let mut buf = vec![];
        proof.serialize_uncompressed(&mut buf).unwrap();
        let proof_size_MB = buf.len() as f64 / (1024.0 * 1024.0);

        // --- Verify ---
        let mut verifier_transcript = ProofTranscript::new(b"example");
        let now = Instant::now();
        BasefoldCommitmentScheme::batch_verify(
            &proof,
            &pp,
            &opening_point,
            &evals,
            &commitments.iter().collect::<Vec<_>>(),
            &mut verifier_transcript,
        )
        .unwrap();
        let verify_ms = now.elapsed().as_millis();

        //
        wtr.write_record(&[
            num_vars.to_string(),
            batch_size.to_string(),
            commit_ms.to_string(),
            prove_ms.to_string(),
            verify_ms.to_string(),
            proof_size_MB.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}


pub fn run_basefold_benchmark_csv(
    output_path: &str,
    num_vars_range: std::ops::RangeInclusive<usize>,
    batch_range: std::ops::RangeInclusive<usize>,
) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(output_path)?;
    wtr.write_record([
        "num_vars",
        "batch_size",
        "commit_ms",
        "prove_ms",
        "verify_ms",
        "proof_size_MB",
    ])?;

    let mut rng = test_rng();

    for &num_vars in num_vars_range.clone().collect::<Vec<_>>().iter() {
        for &batch_size in batch_range.clone().collect::<Vec<_>>().iter() {
            let pp = BasefoldPP::<GoldilocksExt2, GoldilocksExt2, Sha256>::new(BASEFOLD_ADDITIONAL_RATE_BITS);

            //
            let polys: Vec<_> = (0..batch_size)
                .map(|_| {
                    let evals = (0..(1 << num_vars)).map(|_| GoldilocksExt2::random(&mut rng)).collect();
                    DensePolynomial::new(evals)
                })
                .collect();

            let opening_point: Vec<_> = (0..num_vars).map(|_| GoldilocksExt2::random(&mut rng)).collect();
            let evals: Vec<_> = polys.iter().map(|poly| poly.evaluate(&opening_point)).collect();

            // --- Commit ---
            let now = Instant::now();
            let commitments = BasefoldCommitmentScheme::batch_commit_polys(&polys, &pp, BatchType::Big);
            let commit_ms = now.elapsed().as_millis();

            // --- Prove ---
            let mut prover_transcript = ProofTranscript::new(b"example");
            let now = Instant::now();
            let proof = BasefoldCommitmentScheme::batch_prove(
                &polys.iter().collect::<Vec<_>>(),
                &pp,
                &commitments.iter().collect::<Vec<_>>(),
                &opening_point,
                &evals,
                BatchType::Big,
                &mut prover_transcript,
            );
            let prove_ms = now.elapsed().as_millis();

            let mut buf = vec![];
            proof.serialize_uncompressed(&mut buf).unwrap();
            let proof_size_MB = buf.len() as f64 / (1024.0 * 1024.0);

            // --- Verify ---
            let mut verifier_transcript = ProofTranscript::new(b"example");
            let now = Instant::now();
            BasefoldCommitmentScheme::batch_verify(
                &proof,
                &pp,
                &opening_point,
                &evals,
                &commitments.iter().collect::<Vec<_>>(),
                &mut verifier_transcript,
            )
            .unwrap();
            let verify_ms = now.elapsed().as_millis();

            //
            wtr.write_record(&[
                num_vars.to_string(),
                batch_size.to_string(),
                commit_ms.to_string(),
                prove_ms.to_string(),
                verify_ms.to_string(),
                proof_size_MB.to_string(),
            ])?;
        }
    }

    wtr.flush()?;
    Ok(())
}


pub fn run_basefold_ef_benchmark_csv(
    output_path: &str,
    num_vars_range: std::ops::RangeInclusive<usize>,
    batch_range: std::ops::RangeInclusive<usize>,
) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(output_path)?;
    wtr.write_record([
        "num_vars",
        "batch_size",
        "commit_ms",
        "prove_ms",
        "verify_ms",
        "proof_size_bytes",
    ])?;

    let mut rng = test_rng();

    for &num_vars in num_vars_range.clone().collect::<Vec<_>>().iter() {
        for &batch_size in batch_range.clone().collect::<Vec<_>>().iter() {
            let pp = BasefoldPP::<Goldilocks, GoldilocksExt2, Sha256>::new(BASEFOLD_ADDITIONAL_RATE_BITS);

            //
            let polys: Vec<_> = (0..batch_size)
                .map(|_| {
                    let evals = (0..(1 << num_vars)).map(|_| Goldilocks::random(&mut rng)).collect();
                    DensePolynomial::new(evals)
                })
                .collect();

            let opening_point: Vec<_> = (0..num_vars).map(|_| Goldilocks::random(&mut rng)).collect();
            let evals: Vec<_> = polys.iter().map(|poly| poly.evaluate(&opening_point)).collect();

            // --- Commit ---
            let now = Instant::now();
            let commitments = BasefoldCommitmentScheme::batch_commit_polys(&polys, &pp, BatchType::Big);
            let commit_ms = now.elapsed().as_millis();

            // --- Prove ---
            let mut prover_transcript = ProofTranscript::new(b"example");
            let now = Instant::now();
            let proof = BasefoldCommitmentScheme::batch_prove(
                &polys.iter().collect::<Vec<_>>(),
                &pp,
                &commitments.iter().collect::<Vec<_>>(),
                &opening_point,
                &evals,
                BatchType::Big,
                &mut prover_transcript,
            );
            let prove_ms = now.elapsed().as_millis();

            let mut buf = vec![];
            proof.serialize_uncompressed(&mut buf).unwrap();
            let proof_size_MB = buf.len() as f64 / (1024.0 * 1024.0);

            // --- Verify ---
            let mut verifier_transcript = ProofTranscript::new(b"example");
            let now = Instant::now();
            BasefoldCommitmentScheme::batch_verify(
                &proof,
                &pp,
                &opening_point,
                &evals,
                &commitments.iter().collect::<Vec<_>>(),
                &mut verifier_transcript,
            )
            .unwrap();
            let verify_ms = now.elapsed().as_millis();

            // 
            wtr.write_record(&[
                num_vars.to_string(),
                batch_size.to_string(),
                commit_ms.to_string(),
                prove_ms.to_string(),
                verify_ms.to_string(),
                proof_size_MB.to_string(),
            ])?;
        }
    }

    wtr.flush()?;
    Ok(())
}


#[test]
fn generate_basefold_benchmark() {
    run_basefold_benchmark_csv("basefold_26_small_batch.csv", 22..=22, (1<<4)..=(1<<4)).unwrap();
    run_basefold_benchmark_csv("basefold_26_medium_batch.csv", 16..=16, (1<<10)..=(1<<10)).unwrap();
    run_basefold_benchmark_csv("basefold_26_large_batch.csv", 10..=10, (1<<16)..=(1<<16)).unwrap();

    run_basefold_ef_benchmark_csv("basefold_22_ef_small_batch.csv", 18..=18, (1<<4)..=(1<<4)).unwrap();
    run_basefold_ef_benchmark_csv("basefold_22_ef_medium_batch.csv", 16..=16, (1<<8)..=(1<<8)).unwrap();
    run_basefold_ef_benchmark_csv("basefold_22_ef_large_batch.csv", 10..=10, (1<<12)..=(1<<12)).unwrap();
}

#[test]
fn generate_basefold_pack_batch_benchmark() {
    basefold_pack_batch_benchmark("basefold_batch_pack.csv", 26, 6).unwrap();
}

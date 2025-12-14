# M&M: Secure Two-Party Machine Learning through Efficient Modulus Conversion and Mixed-Mode Protocols
This repo contains a proof-of-concept implementation for our paper [M&M](https://eprint.iacr.org/2025/1551).


### Repo Directory Description
- `include/` Contains implementation of M&M's linear protocols.
- `SCI/` A fork of CryptFlow2's SCI library and contains implementation of M&M's proposed protocols.
- `networks/` Auto-generated cpp programs that evaluate some neural networks.
- `pretrained/` Pretrained neural networks and inputs.
- `patch/` Patches applied to the dependent libraries.
- `credits/` Licenses of the dependencies. 
- `scripts/` Helper scripts used to build the programs in this repo.

### Requirements

* openssl 
* c++ compiler (>= 8.0 for the better performance on AVX512)
* cmake >= 3.13
* git
* make
* OpenMP (optional, only needed by CryptFlow2 for multi-threading)

### Building Dependencies
* Run `bash scripts/build-deps.sh` which will build the following dependencies
	* [emp-tool](https://github.com/emp-toolkit/emp-tool) We follow the implementation in SCI that using emp-tool for network io and pseudo random generator.
	* [emp-ot](https://github.com/emp-toolkit/emp-ot) We use Ferret in emp-ot as our VOLE-style OT.
	* [Eigen](https://github.com/libigl/eigen) We use Eigen for tensor operations.
	* [SEAL](https://github.com/microsoft/SEAL) We use SEAL's implementation for the BFV homomorphic encryption scheme.
	* [zstd](https://github.com/facebook/zstd) We use zstd for compressing the ciphertext in SEAL which can be replaced by any other compression library.
	* [hexl](https://github.com/intel/hexl/tree/1.2.2) We need hexl's AVX512 acceleration for achieving the reported numbers in our paper.

* The generated objects are placed in the `build/deps/` folder.
* Build has passed on the following setting
  * MacOS 11.6 with clang 13.0.0, Intel Core i5, cmake 3.22.1
  * Red Hat 7.2.0 with gcc 7.2.1, Intel(R) Xeon(R), cmake 3.12.0
  * Ubuntu 18.04 with gcc 7.5.0 Intel(R) Xeon(R),  cmake 3.13
  * Ubuntu 20.04 with gcc 9.4.0 Intel(R) Xeon(R),  cmake 3.16.3
  
### Building Demo

* Run `bash scripts/build.sh` which will build 6 executables in the `build/bin` folder
	* `resnet50` 
	* `sqnet`
	* `densenet121`

### Run Demo 

<!-- 1. Run `bash scripts/run-server.sh sqnet`. The program will load the pretrained model in the folder `pretrained/` which might takes some time when the pretrained model is huge. 

2. On other terminal run `bash scripts/run-client.sh sqnet`. The program will  load the prepared input image in the folder `pretrained`.  
   * replace `sqnet` by `resnet50` to run on the ResNet50 model.
   * replace `sqnet` by `densenet121` to run on the DenseNet121 model. -->

   * Run `bash scripts/sqnet-run.sh` to run on the SqueezeNet model.
   * Run `resnet-run.sh` to run on the ResNet50 model.
   * Run `densenet-run.sh` to run on the DenseNet121 model.

   The program will invoke two process, a client and a server.

   You can change the `SERVER_IP` and `SERVER_PORT` defined in the [scripts/common.sh](scripts/common.sh) to run the demo remotely.
Also, you can use our throttle script to mimic a remote network condition within one Linux machine, see below.

### Micro-Benchmarks
  1. Run `scripts/ring_to_field_test-run.sh bw` to run the conversion of ring to field. Current supported bitwidth : {16, 32, 40, 48, 56, 60, 64}.

  2. Run `scripts/field_to_ring_test-run.sh bw` to run the conversion of field to ring. Current supported bitwidth : {16, 32, 40, 48, 56, 60, 64}.

  3. Run `scripts/ring_extension_test-run.sh bw` to run the ring extension. Base ring = $2^{64}$, target ring = $2^{64 + \text{bw}}$ Current supported lift-bitwidth : [1, 64].

  4. Run `scripts/bole_test-run.sh [flag0] [flag1]` to run Batch Oblivious Linear Evaluation. 
  * Base mod and target mod are $2^l$, itermedia mod is a NTT-friendly prime (enable SIMD encode in FHE).
  * The procedure consists a Ring-to-Field conversion, a FHE SIMD based element-wise product, a Field-to-Ring conversion (with or without free truncation).
  * set `flag0` to `0` : element-wise product of a secret share vector with a private vector (hold by Server)
  * set `flag0` to `1` : element-wise product of two secret share vectors
  * set `flag1` to `0` : preform exactly field-to-ring convertion
  * set `flag1` to `1` : perform free truncate while field-to-ring convertion


### Mimic an WAN setting within LAN on Linux

* To use the throttle script under [scripts/throttle.sh](scripts/throttle.sh) to limit the network speed and ping latency (require `sudo`)
* For example, run `sudo scripts/throttle.sh wan` on a Linux OS which will limit the local-loop interface to about 1Gbps bandwidth and 40ms ping latency.
  You can check the ping latency by just `ping 127.0.0.1`. The bandwidth can be check using extra `iperf` command.
<!-- * We pre-defined 3 network to 

| Network | Bandwidth | Ping Latency |
| ---- | ---- | ---- |
| LAN  | 1Gbps | 2ms |
| WAN  | 1Gbps | 40ms |
| MAN  | 100Mbps | 40ms | -->

### Citing

```text
@misc{cryptoeprint:2025/1551,
      author = {Ye Dong and Wen-jie Lu and Xiaoyang Hou and Kang Yang and Jian Liu},
      title = {M&M: Secure Two-Party Machine Learning through Efficient Modulus Conversion and Mixed-Mode Protocols (Full Version)},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/1551},
      year = {2025},
      url = {https://eprint.iacr.org/2025/1551}
}
````

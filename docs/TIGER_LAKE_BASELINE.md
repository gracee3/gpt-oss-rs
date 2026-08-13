# Tiger Lake Optimization Baseline

The implementation sprint began at
`debb9d74d01af5f78deefc013364aba1129b49c1` on
`agent/tiger-lake-optimization-foundation`. The remote branch matched that
commit and `origin/main` remained at `6caf274`. The only pre-existing local
file was the now-checked-in historical preplanning document.

The post-upgrade host is Ubuntu 26.04 LTS on kernel `7.0.0-29-generic` with
microcode `0xbe`. Model-free CPUID reports GenuineIntel family 6, model 140,
stepping 1, eight logical CPUs, OSXSAVE, and XCR0 `0x2e7`. The machine has
four physical cores, 31 GiB usable RAM, and 188 GiB free at sprint start.

Iris Xe remains `8086:9a49` with Dell subsystem `1028:0a42`, driven by i915.
The current OpenCL stack reports Intel NEO `26.05.037020`, OpenCL 3.0,
SPIR-V through 1.5, subgroup and integer-dot support, unified memory, SVM,
and Intel USM. The loader/driver/IGC hashes were recaptured. Level Zero loader
1.28.2 is installed; the repository research build still requires the absent
pinned header corpus, so Level Zero remains environment evidence rather than
a production dependency.

Raw evidence is external at
`/home/emmy/gpt-oss-rs-artifacts/tiger-lake-optimization/debb9d74d01af5f78deefc013364aba1129b49c1/baseline/`.
Its `SHA256SUMS` index hash is
`a07a167fad4cbd3d918f303b1d0cf85018779e574c0936eff4b98893a9511ea4`.
The checked-in oracle image inputs and lock remain byte-identical to
`af6c0a2`; this baseline does not recertify that historical candidate.

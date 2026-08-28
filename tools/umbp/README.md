# UMBP deployment tools

`bootstrap_distributed_node.sh` prepares the reproducible vLLM, MoRI, and
llm-d Router software environment for UMBP validation. It builds MoRI with
UMBP enabled and SPDK disabled:

```text
BUILD_UMBP=ON
BUILD_UMBP_SPDK=OFF
```

## Why the bootstrap does not enable SPDK

NVMe hardware being visible does not mean that a device is safe to give to
SPDK. SPDK takes exclusive userspace control of a PCI controller. Preparing a
device normally unbinds it from the Linux `nvme` driver and may allow UMBP to
overwrite its contents with raw tier data. This can make existing data or a
device used by another workload inaccessible.

At the time this environment was inspected, the host exposed eight 3.5-TB
NVMe namespace devices, `/dev/nvme0n1` through `/dev/nvme7n1`. None had a
filesystem or mount point reported by `lsblk`, and 18,432 free 2-MiB hugepages
were available. However, all eight controllers remained bound to the kernel
`nvme` driver, SPDK libraries were not installed, and no device had been
explicitly approved for destructive raw use. An empty mount-point column alone
does not prove that a disk is unused: it may still contain signatures or data,
belong to LVM or RAID, have holders or open handles, or be reserved by the host
operator.

The bootstrap therefore stops at the safe software baseline. It does not:

- choose, erase, format, bind, or unbind an NVMe device;
- install or configure a host SPDK/DPDK runtime;
- change hugepage allocation, memory-lock limits, or container privileges; or
- change firewall, RDMA, IOMMU, or VFIO configuration.

This is a safety boundary, not an AMD or UMBP hardware limitation. DRAM and
RDMA UMBP correctness can be tested with this build, but the SPDK/NVMe tier
cannot.

## Enabling SPDK deliberately

Before rebuilding with SPDK, an operator must name one disposable NVMe device
and approve it for exclusive raw use. Record and verify both its stable serial
number and PCI address; `/dev/nvmeXn1` names can change after reboot. Then check
its filesystem signatures, partitions, mounts, swap use, LVM/RAID membership,
device holders, and open handles before changing its driver binding.

Once that validation and approval are complete, the intended Mori build is:

```bash
BUILD_UMBP=ON BUILD_UMBP_SPDK=ON \
  uv pip install --no-build-isolation --editable /app/umbp/moriv112
```

`BUILD_UMBP_SPDK=ON` causes Mori's Python build to prepare SPDK and configures
CMake with `USE_SPDK=ON`. A successful build is only the first step: the
approved controller must also be bound and passed to Mori through its PCI and
SSD configuration. Run Mori's SPDK preflight before any UMBP test, followed by
a small raw-I/O correctness test, a forced SSD restore correctness test, and
only then the tier-2 performance comparison.

Do not copy an example PCI address from another host. Resolve and approve the
device independently on every machine.

## Checking two-node RDMA connectivity

`check_rdma_peer.sh` contains the interface and address maps for local host
`192.168.0.69` and remote host `192.168.0.185`. From the local host, check the
management route and all eight routed RoCE IPv6 paths:

```bash
tools/umbp/check_rdma_peer.sh ip
```

The two hosts use different routed `/64` prefixes, so the script verifies the
selected egress interface instead of requiring identical prefixes. ICMP only
proves IP connectivity. To test an actual RDMA read, first use `show_gids` on
each host to find the GID index for `ionic_0`. On the remote host, run:

```bash
tools/umbp/check_rdma_peer.sh rdma-server 0 <remote-gid-index>
```

Then, on the local host, run:

```bash
tools/umbp/check_rdma_peer.sh rdma-client 0 <local-gid-index>
```

The script passes both `--ipv6` and `--ipv6-addr` to `ib_read_bw`. The first
selects an IPv6 GID and the second makes perftest use IPv6 for its TCP parameter
exchange. Omitting `--ipv6-addr` makes perftest resolve the IPv6 peer with
`AF_INET` and fail before it creates the RDMA connection.

The link index ranges from `0` to `7` and selects the corresponding
`ionic_<index>` device. GID indexes can differ between hosts and must be
resolved independently. The `ip` and `ping` commands are provided by
`iproute2` and `iputils-ping`; `ib_read_bw` is provided by `perftest`.

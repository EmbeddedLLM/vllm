# UMBP two-node RDMA stress report

## Scope

This report records the two-node RDMA validation performed on 2026-08-28
between:

- client: `192.168.0.69`;
- server: `192.168.0.185`; and
- RDMA devices: `ionic_0` through `ionic_7`.

The test used `tools/umbp/stress_rdma_links.sh` with its default configuration:

```text
MODE=both
DURATION=60
MESSAGE_SIZE=1048576
GID_INDEXES=1,1,1,1,1,1,1,1
```

Each Ionic device ran one RDMA read and one RDMA write process concurrently.
Every process used a distinct TCP parameter-exchange port and global `fc01:`
RoCE v2 GID index `1`. Results came from the client logs in
`rdma-stress-logs/`.

## Hardware state

All eight client interfaces reported:

```text
Link speed: 400000 Mb/s
Duplex: full
Link state: active
RDMA MTU: 4096 bytes
PCIe link: 32 GT/s x16
```

`ionic_0` through `ionic_3` are attached to NUMA node 0. `ionic_4` through
`ionic_7` are attached to NUMA node 1. The host has two NUMA nodes and 236
online logical CPUs.

An `ethtool` inspection immediately after the run found no nonzero counters
whose names contained error, drop, discard, timeout, retry, or fault. This is
useful negative evidence, but it is not a synchronized before-and-after
counter capture and does not replace examination of the remote counters.

## Results

| Device | Read (Gbit/s) | Write (Gbit/s) | Combined (Gbit/s) |
| --- | ---: | ---: | ---: |
| `ionic_0` | 190.99 | 195.18 | 386.17 |
| `ionic_1` | 45.69 | 269.83 | 315.52 |
| `ionic_2` | 238.08 | 179.69 | 417.77 |
| `ionic_3` | 244.20 | 215.16 | 459.36 |
| `ionic_4` | 213.06 | 29.69 | 242.75 |
| `ionic_5` | 215.26 | 29.60 | 244.86 |
| `ionic_6` | 226.49 | 30.32 | 256.81 |
| `ionic_7` | 224.19 | 30.60 | 254.79 |
| **Total** | **1,597.96** | **980.07** | **2,578.03** |

The totals correspond to:

```text
Read payload bandwidth:      199.75 GB/s
Write payload bandwidth:     122.51 GB/s
Combined payload bandwidth:  322.25 GB/s
```

All 16 tests completed without a connection or Queue Pair transition failure.
The logs report Ethernet link layer, RC transport, 4,096-byte MTU, and global
RoCE v2 GID index `1`.

## Theoretical comparison

Eight full-duplex 400-Gbit/s links provide a nominal line rate of:

```text
One direction:  3,200 Gbit/s
Full duplex:    6,400 Gbit/s
```

Relative to those nominal rates, the concurrent test reached:

```text
Read direction:    49.9%
Write direction:   30.6%
Combined duplex:   40.3%
```

These percentages compare application payload throughput with physical line
rate, so protocol overhead prevents exactly 100% payload efficiency. That
overhead is not large enough to explain the measured gap.

The earlier isolated `ionic_0` correctness run reached 245.73 Gbit/s using
65,536-byte transfers and 1,000 iterations. That short run proves functional
remote RDMA reads but is not long enough to establish sustainable peak
throughput.

## Assessment

The run proves that all eight routed RoCE v2 paths can establish an RC Queue
Pair and transfer data. There is no current evidence of a disconnected link,
an interface negotiated below 400 Gbit/s, or a PCIe link negotiated below Gen5
x16.

The run does not demonstrate theoretical-limit performance. Its strongest
anomalies are:

- `ionic_1` read throughput of only 45.69 Gbit/s while its write throughput is
  269.83 Gbit/s; and
- `ionic_4` through `ionic_7` write throughput clustered between 29.60 and
  30.60 Gbit/s while their reads remained between 213.06 and 226.49 Gbit/s.

The common low-write behavior across all four NUMA-node-1 devices points first
to shared CPU scheduling, memory placement, PCIe-root, remote-host resource, or
fabric Quality-of-Service contention rather than four independent physical
link failures. The `MODE=both` run launches 16 unpinned processes and is
intentionally an aggressive system stress test. It is not a controlled
single-direction peak-bandwidth benchmark.

The current evidence is therefore sufficient to call every link functional,
but insufficient to call every link performance-healthy. Direction-isolated
and NUMA-controlled runs are required before attributing the anomalies to the
NICs or network.

## Follow-up validation

Run read-only traffic on both hosts with identical settings:

```bash
DURATION=120 MODE=read tools/umbp/stress_rdma_links.sh server
DURATION=120 MODE=read tools/umbp/stress_rdma_links.sh client
```

Preserve the client and server logs as `rdma-stress-logs-read`, then run the
write-only test:

```bash
DURATION=120 MODE=write tools/umbp/stress_rdma_links.sh server
DURATION=120 MODE=write tools/umbp/stress_rdma_links.sh client
```

Preserve those logs as `rdma-stress-logs-write`. The server command must start
before its matching client command.

Interpret the isolated results as follows:

- If most links reach approximately 300 to 380 Gbit/s, the physical paths are
  likely healthy and the original result primarily reflects simultaneous
  system contention.
- If `ionic_1` remains near 45 Gbit/s during isolated reads, investigate that
  path independently.
- If `ionic_4` through `ionic_7` remain near 30 Gbit/s during isolated writes,
  investigate NUMA-node-1 placement, the matching remote topology, and switch
  Quality-of-Service configuration.
- If NUMA-pinned processes remove the anomaly, classify it as host placement
  rather than a network failure.

For a conclusive diagnosis, capture client and server NIC counters immediately
before and after each run. Retain both hosts' perftest logs and record CPU and
memory affinity. A future harness revision should add per-device NUMA binding
and synchronized counter deltas so the isolated and concurrent results are
directly comparable.

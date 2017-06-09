---
layout: post
title: What is a Multiraft?
date: 2017-06-09 17:50 +0300
comments: true
---

In computer science there is a well known problem called _"consensus"_. In a nutshell, it's a task of getting all participants in a group to agree on some specific value based on the votes of each member. There are also several algorithms that aim to solve this problem, namely [Paxos][paxos-wiki], [Raft][raft-wiki], [Zab][zab], [2PC][2pc-wiki]. What is Multiraft then?

# Consensus problem

Let's first start with some problem statement and definition. Again, [wikipedia][consensus-wiki] offers quite nice introduction to what it is and some other relevant information. In short (and using my own words), consensus problem arises in a distributed settings, where multiple processes are present, they communicate over some faulty medium (while also being faulty by themselfes), and single decision should be chosen. Usually, people say that multiple processes have to agree on some value, or it could be agreement over commiting update to a database or not, or select a leader in a group, etc.

One of obstacles for convinient problem solution is a (pretty high) probabilty of a failure. Communication medium could be not that reliable, messages could get lost and misinterpreted, group members could also fail or just misbehave. Another obstacle is asyncronous nature of a system: one can't be sure the speed of message delivery is constant, so if a message takes too long to arrive, it's very hard to distinguish if something bad happened or is it just late.

Just a side note, strict consensus problem theory covers much more and deals with more diverse constraints-limitations and assumptions combinations. I won't cover them in this post.

# Raft

![Raft logo](https://raft.github.io/logo/annie-solo.png){:height="200x"}

Raft is one of the algorithm to solve consensus problem that aims to be easy to understand and to implement. Before going into further details I highly recommend walking through [this nice visualization](http://thesecretlivesofdata.com/raft/).

Just to reiterate, Raft solves problem by electing a leader and doing everything through it. If the leader fails, new leader is elected. If client asks non-leader about something, he is redirected to the leader. On update arrival leader disseminate the change through the group and ensures change persists. Simple, right?

Those interested in technical details and more information are welcome to visit ["original" website][raft]. Also, [original paper](https://ramcloud.stanford.edu/wiki/download/attachments/11370504/raft.pdf) describes how to implement the algorithm. However, devil is in details.

# etcd, CockroachDB and TiDB

Both [CockroachDB](https://github.com/cockroachdb/cockroach) and [TiDB](https://github.com/pingcap/tidb) are distributed, transactional, consistent SQL databases, while [etcd](https://github.com/coreos/etcd) is a reliable, distributed key-value store. These distributed systems use Raft as their consensus algorithm. Both databases Raft algorithm implementations are based on [etcd raft library](https://github.com/coreos/etcd/tree/master/raft).

CockroachDB uses it to agree on some objects: namely, `Range`'s that holds some data (key-value pairs for some range of keys). Each node holds multiple ranges and therefore participates in multiple Raft groups. Nothing wrong with that, basically.

In a blog post [Scaling Raft](https://www.cockroachlabs.com/blog/scaling-raft/) CockroachLabs describes the problem and their solution: "modify" raft to handle not single value (that is, `Range`), but multiple of them. So, MultiRaft is used.

Here is visualization of differencies between original Raft and MultiRaft (images taken from original CockroachLabs blog post), left is original, right is MultiRaft:

![Vanilla Raft](https://www.cockroachlabs.com/uploads/2015/06/multinode2-300x216.png){:style="margin-left:50px"}
![MultiRaft](https://www.cockroachlabs.com/uploads/2015/06/multinode3-300x212.png){:style="float: right; margin-right:50px"}

Obvious difference here is number of connections and messages that are used to make everything work. Multiraft works by effectively "multiplexing" many communcation channels into per-node ones.

As per more detailed description, problems begin when number of ranges per node (replica) increases. Obviously, then node has to participate in many-many raft groups with all that possible overhead. I haven't stated that previously, but raft protocol assumes periodic heartbeat events to be exchanged within the group. What does happen if one node is a member of multpiple groups? Service protocol traffic increases. Using MultiRaft (and using only one Raft instance per Node  - `Store` in Cockroach source code terminology - rather than per `Range`) solves the problem of traffic increase. Under the hood, CockroachDB coalesces/uncoalesces heartbeats in a sinlge response/request (according to `pkg/storage/store.go`).

On the other hand, TiKV (as a underlying building block for TiDB) also uses MultiRaft for exactly the same purpose. Basically, its atomic unit of transfer is called `Region` (rather than `Range`). Behind the scenes, it uses multiplexion for Region-containing messages but no heartbeat coalescion. The best explanation I've found is [here](https://groups.google.com/forum/#!topic/tidb-user/uQXlBnTxsoI).

### References:

 * <https://www.cs.rutgers.edu/~pxk/417/notes/content/consensus.html>
 * <https://www3.cs.stonybrook.edu/~prade/Teaching/F15-CSE535/prez/L18-L19-L20-Consensus.pdf>
 * <https://github.com/pingcap/blog-cn/blob/master/the-design-and-implementation-of-multi-raft.md>


[paxos-wiki]: https://en.wikipedia.org/wiki/Paxos_(computer_science)
[raft-wiki]: https://en.wikipedia.org/wiki/Raft_(computer_science)
[zab]: http://zookeeper.apache.org/doc/r3.3.6/zookeeperInternals.html
[consensus-wiki]: https://en.wikipedia.org/wiki/Consensus_(computer_science)
[2pc-wiki]: https://en.wikipedia.org/wiki/Two-phase_commit_protocol
[raft]: https://raft.github.io/

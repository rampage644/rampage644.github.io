---
layout: post
title: Consistency, causal and eventual
date: 2017-06-26 17:30 +0300
comments: false
---

In some previous [blog post]({{ site.baseurl }}{% post_url 2017-06-14-transactions-consistency %}) several consistency models were briefly described as well as what do they mean in terms of database transactions. With following writing, I would like to further remove white knowledge gap and explain in simple terms some other well-known consistency models, such as _eventual_ and _causal_. These are much more common within distributed systems rather than traditional databases, though, in the present time there is little difference between the two.

# Sequential consistency

Let's recall that last time we've coveded consistency models that followed right path:

![aphyr consistency models](https://aphyr.com/data/posts/313/family-tree.jpg){:height="400px"}

So, we'll assume that _serializable_ model descendants are not that scary as they are for the first time. Today, _sequential_ consistency model (left path of the image) lineage would be unveiled.

But first, let's make ourselves familiar and comfortable with _sequential_ model itself. It was defined by Lamport as a property:

> the result of any execution is the same as if the reads and writes occurred in some order, and the operations of each individual processor appear in this sequence in the order specified by its program.

Well, I'd like to have more intuitve explanation. Sure, if one devotes enough mental efforts this statement could be easily deciphered, but humans are inherently lazy (at least I am), so let's try to rewrite its definition (or better, redraw!) for easier understanding.

Different resources on the web use different examples to explain sequential consistent systems. Here, I want to emphasize that it's sufficient to imageine some abstract clients (independent) and some shared resource they all have access to. For example, CPUs (or threads) could be our clients and RAM (or register) could be shared resource. Also, some disributed system (say, HDFS) could be shared resource and Alice and Bob applications could be our clients. But it's basically just some independent workers and shared resource.

Also, all that consistency models are just fancy way to describe a contract between system and its users: what users can expect of the system during interaction, what state does system have to be in.

Okay, let's imagine there are tho processes (`P1` and `P2`) who talk to a system. First process does read read `R(x)`, then blue write `W(x)` and green write `W(y)`. Second process performs red read `R(y)`, then blue read `R(x)`, then green write `W(y)`.

If our system claims to be strictly serializable, then it has to appear as single global place with all operations appear to _all_ processes exactly in their global order. Probably, such system would need global wall clock time (which is pretty hard to obtain).

![Sequential part1]({{site.url}}/assets/eventual_consistency/sequential_p1.png)

What does system guarantee? It says process's `P1` red read happens before its blue write, and its blue write happens before green one. It also says that process's `P2` red read happen before process `P1` red read, i.e. all events are totally ordered. Pretty strong guarantee.

Now let's imagine systems says it's only sequential consistent. What does change? Well, it still guarantees that `P1` process read happen before its writes, and blue write occurs before green. The same is true for the process `P2`. So, every process's _operations order is preserved_. But subtle difference is that total order is not guaranteed: system may interleave operations as it likes:

![Sequential part2]({{site.url}}/assets/eventual_consistency/sequential_p2.png)

Every process operation order is preserved (as per definition), but _real_ order is not guaranteed. Some operations may even look like to happen in the past! So, sequential consistency says every process that its operations would appear in order they're issued. No global clock requirement, though.

# References

 * <http://www.news.cs.nyu.edu/~jinyang/fa09/notes/ds-consistency.pdf>

---
layout: post
title: Transactions and consistency 101
date: 2017-06-14 13:00 +0300
mathjax: True
comments: True
---

For a very long time, there were some white spots about transactions and different consistency models. Don't get it wrong, my bet is that every engineer has some _intrinsic_ understanding of what transaction is, knows that it somehow related to databases, that consistency has something to do with a system being in a predictable and understandable state. But being a complete self-learner, one has to skip some theoretical things and start doing practical things as soon as possible.

# Transactions

It's well known that relational DBMS are all about [Relational Algebra](https://en.wikipedia.org/wiki/Relational_algebra) and [Entity–relationship model](https://en.wikipedia.org/wiki/Entity%E2%80%93relationship_model). In other words, ER modeling is used to represent different things in a business to perform its process and operate successfully; it is abstract data model. RDMBS are systems invented to operate on business data; they store data that is modeled with Entity-Relationship model and perform operations (queries, updates) on it.

However, there are some discrepancies of what is natural for human and for a machine. The most simple example is reading. For a man, it's natural to think about both single letter as a unit, single word, and even single sentence. _Depending on representation_, single letter is atomic for a machine, but words and sentences are not atomic but are sequences of others objects (let's not consider [word2vec representations](https://en.wikipedia.org/wiki/Word2vec) for now).

With ER modeling introduced some natural business operations are atomic in human mind but not for a machine. The classic example is money transfer: for us (people) it single/indivisible operation. But from the machine point of view (again, depends on representation, i.e. modeling) it could be comprised of a sequence of actions. _Transaction_ then is a way to enforce high-level abstractions and correctness on an inherently low-level system.

To quote Stanford CS145 class:
> A transaction (“TXN”) is a sequence of one or more operations (reads or writes) which reflects a single real world transition.

But keeping system state consistent (_"correct"_) is not the only motivation for transactions: another one is concurrency that allows much better performance.

Traditionally, one speaks about ACID properties of a transaction: _atomicity_, _consistency_, _isolation_ and _durability_. To make the story complete, atomicity requires a transaction to either happen or not, durability states change made by transaction persist. But here we will focus on isolation and consistency.

# Consistency models

First, let's start with some properties. On one hand, there is __linearizability__ that has nothing to do with transactions but rather with a single global state (or just single object). It is a real-time guarantee that all operations appear instantaneous: no overlap, no interference. Once you write something, subsequent reads return that _"something"_. For example, the hardware provides atomic compare-and-swap operation.

On the other hand, there is a property called __serializability__. This relates to transactions. Basically, this is a guarantee that a set of transactions over multiple objects _is_ an equivalent of some serial (that is, as each transaction applied at a time - total ordering) execution. Consider two transactions that themselves are consistently executed simultaneously:

![Serial execution #1]({{site.url}}/assets/transactions_consistency/2017-06-14-103235_682x234_scrot.png)

Here (sample courtesy of CS145 2016) first transaction (red) transfers $100 from account A to account B. Blue transactions applies 6% interest rate to both of them.

This execution order (interleaved, both transactions happen simultaneously) is not equivalent to serial execution (red, then blue or blue, then red).

![Interleaved execution #1]({{site.url}}/assets/transactions_consistency/2017-06-14-103258_762x246_scrot.png)

However, such interleave __is__ equivalent to "red, then blue" order. Hence, this set of transactions has a property of _serializability_.

![Interleaved execution #2]({{site.url}}/assets/transactions_consistency/2017-06-14-103252_765x225_scrot.png)

Note that this guarantee says nothing about single operations order (no real-time constraints as for _linearizability_). All it assumes is that there exists some serial execution order for transactions.

Combining the two we get __strict (strong) serializabilty__. For quick explanation let me just quote wiki (emphasis mine):
> This is the _most rigid model_ and is impossible to implement without _forgoing performance_. In this model, the programmer’s expected result will be received every time. _It is deterministic_.

I wish I had time to describe models that are _"descendants"_ of sequential consistency. However, this post is mainly about transactions. So let's continue with "serializable" ones. We're following the right path from this image:

![aphyr consistency models](https://aphyr.com/data/posts/313/family-tree.jpg){:height="400px"}

Okay, back to transactions then. Serializability is a useful concept because it allows programmers to ignore issues
related to concurrency when they code transactions. But it seems that:
> To complicate matters further, what most SQL databases term the SERIALIZABLE consistency level actually means something weaker, like _repeatable read_, _cursor stability_, or _snapshot isolation_.

By weakening our models we improve performance but it requires additional effort from a programmer to maintain correctness. Let's explain those _RR_, _SI_, _CS_ and _RC_ abbreviations.

As a reference, I'd like to use [Morning paper](https://blog.acolyer.org/2016/02/24/a-critique-of-ansi-sql-isolation-levels/) overview of [Critique of ANSI SQL isolation levels](https://arxiv.org/pdf/cs/0701157.pdf). Really, I can't help being overly excited by Adrian Colyer job was done in his field! So, for very brief overview one can follow this post, more details are available in "Morning Paper" post, and even more, technical details are available in the paper itself.

ANSI SQL defines different isolation levels and phenomena (well, the standard doesn't define those actually). Different isolation levels eliminate different phenomenon (anomalies). Let's start with the bottom (images taken from "Morning paper" blog post).

_Dirty write_ occurs when one transaction overwrites value written but not yet committed by another.
![Dirty write](https://adriancolyer.files.wordpress.com/2016/02/dirty-write.png?h=300)

_Dirty read_ occurs when one transaction reads a value that was written by the uncommitted transaction. If latter rollbacks, former ends up with inconsistent value.
![Dirty read](https://adriancolyer.files.wordpress.com/2016/02/dirty-read.png?h=300)

_Read Committed (RC)_ isolation level eliminates these phenomena.

_Lost update_ happens when two concurrent transactions read and then update (probably based on a value they read) the same item.
![Lost update](https://adriancolyer.files.wordpress.com/2016/02/lost-update.png?h=300).

_Cursor Stability (CS)_ eliminates it (for additional details consult the post and paper, this is not quite correct just simple to understand).

_Fuzzy read_ happens when the first transaction subsequent reads return different result due to the second transaction modifies item accessed.
![Fuzzy read](https://adriancolyer.files.wordpress.com/2016/02/fuzzy-read.png?h=300)

_Read skew_ and _write skew_ are a database constrain violations. Consider following integrity constrains: $$ x + y == 100 $$ for the first image and $$ x + y \leq 100 $$ for the second:

![Read skew](https://adriancolyer.files.wordpress.com/2016/02/read-skew.png?h=300)
![Write skew](https://adriancolyer.files.wordpress.com/2016/02/write-skew1.png?h=300)

_Repetable Read (RR)_ eliminates those three.

_Snapshot Isolation_, like _RR_, is stronger level than _RC_ but weaker than _Serializable_. It belongs to MVCC (Multiversion concurrency control) family. In snapshot isolation, we can imagine that each transaction is given its own version, or snapshot, of the database when it begins. When the transaction is ready to commit, it's aborted if a conflict is present.

However, _SI_ is incomparable to _RR_ since it permits _Write skew_ and _RR_ permits at least some phantoms. Wait, what phantom is?

The _Phantom_ phenomenon occurs when we do some predicate-based query (`SELECT WHERE predicate`) and another transaction modifies the data matched with predicate so the subsequent query returns a different result.
![Phantom](https://adriancolyer.files.wordpress.com/2016/02/phantom.png?w=600)

And, finally, no phenomena described are allowed at a _serializable_ level.

# Closing thoughts

Of course, this is only a beginning since there much more consistency models to investigate (left branch of our map) that a better described with a distributed systems in mind. I wish I write the second part blog post in some distant future.

# References

 * <http://web.stanford.edu/class/cs145/>
 * <http://www.bailis.org/blog/linearizability-versus-serializability/>
 * <http://www.bailis.org/blog/when-is-acid-acid-rarely/>
 * <https://blog.acolyer.org/2016/02/24/a-critique-of-ansi-sql-isolation-levels/>
 * <https://github.com/aphyr/distsys-class>
 * <https://aphyr.com/posts/313-strong-consistency-models>
 * <http://www.vldb.org/pvldb/vol7/p181-bailis.pdf>
 * <http://publications.csail.mit.edu/lcs/pubs/pdf/MIT-LCS-TR-786.pdf>
 * [Distributed Systems Concepts and Design](http://www.cdk5.net/wp/)
 * [Database System Concepts](http://db-book.com/)

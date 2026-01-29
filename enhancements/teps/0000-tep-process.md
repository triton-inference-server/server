<!--
# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->
# Triton Enhancement Proposals

**Status**: Draft

**Authors**: [whoisj](https://github.com/whoisj)

**Category**: Process

**Replaces**: N/A

**Replaced By**: N/A

**Sponsor**: [whoisj](https://github.com/whoisj)

**Required Reviewers**: [dzier](https://github.com/dzier), [nnshah1](https://github.com/nnshah1)

**Review Date**: 17 Oct 2025

**Pull Request**: [N/A](https://github.com/triton-inference-server/server/pull/8517)

## Summary

A standard process and format for proposing and capturing architecture, design, and process decisions for the Triton project along with the motivations behind those decisions.
We adopt a similar process as adopted by Dynamo, Kubernetes, Rust, Python, and Ray broadly categorized as "enhancement proposals".

## Motivation

With any software project but especially agile, open source projects in the AI space, architecture, design, and process decisions are made rapidly and for specific reasons which can sometimes be difficult to understand after the fact.
For Triton in particular many teams and community members are collaborating for the first time and have varied backgrounds and design philosophies.
The Triton project's code base itself reflects multiple previously independent code bases integrated quickly to meet overall project goals.
As the project evolves we need a way to propose, ratify and capture architecture, design and process decisions quickly and thoughtfully in a transparent, consistent, lightweight, maintainable way.

Borrowing from the motivation for KEPs:

> The purpose of the KEP process is to reduce the amount of "tribal knowledge" in our community.
> By moving decisions from a smattering of mailing lists, video calls and hallway conversations into a well tracked artifact, this process aims to enhance communication and discoverability.

### Goals

* **Useful**

  Enhancement proposals and the process of writing and approving them should encourage the thoughtful evaluation of design, process, and architecture choices and lead to timely decisions with a clear record of what was decided, why, and what other options were considered.

* **Lightweight and Scalable**

  The format and process should be applicable both to small or medium sized changes as well as large ones.
  The process should not impede the rate of progress but serve to provide timely feedback, discussion, and ratification on key proposals.
  The process should also support retroactive documents to capture and explain decisions already made.

* **Single Document for Requirements and Design**

  Combine aspects of requirements documents, design documents and software architecture documents into a single document.
  Give one place to understand the motivation, requirements, and design of a feature or process.

* **Support Process, Architecture and Guideline Decisions**

  Have a single format to articulate decisions that effect process (such as github merge rules or templates) as well as code and design guidelines as well as features.

* **Clear**

  Should be relatively clear when a document is required, when the review needs to be completed, and by who and what the overall process is.

* **Encourage Collaboration**

  Should allow for easy collaboration and communication between *Authors** and **Reviewers**.

* **Flexible**

  Format and process should be flexible enough to be used for different types of decisions requiring different levels of detail and formatting of sections.

### Non Goals

* Triton Enhancement Proposals (TEP)s do not take the place of other forms of documentation such as user / developer facing documentation (including architecture documents, api documentation)
* Prototyping and early development are not gated by design / architectural approval.
* TEPs should not be a perfunctory process but lead to discussion and thought process around good designs.
* Not all changes (bug fixes, documentation improvements) need a TEP - and many can be reviewed via that normal GitHub pull request

## Proposal

Following successful open source projects such as Kubernetes (KEP) and Dynamo (DEP) we adopt a markdown based enhancement proposal format designed to support any decisions we need to capture as a project.
We will adopt an open, community-wide, discussion and comment process using pull requests but enable **Code-Owners** and **Maintainers** to be the final arbiters of **Approval**.

Subject area experts will be listed as required **Reviewers** to ensure proposals are complete and reviewed properly.

<!-- Enhancement proposals will be stored in github in a separate repository. -->

We provide two templates "limited" and "complete" where the limited template is a strict subset of the complete template, and both indicate which sections are required and which are optional.

## Implementation Details

### Proposal Process

<!-- * Fork or create a branch in the `enhancements` repository -->

* Copy the [limited template](../NNNN-template-limited.md) or [complete template](../NNNN-template-complete.md) to `teps/NNNN-my-feature.md` (where `my-feature` is descriptive, don't assign an `TEP` identifier yet)

  > [!Note]
  > Choose the template that fits your purpose.
  > You can start with the limited form and pull additional sections from the complete form as needed.
  > Keep the order of the sections consistent.

* Identify a **Sponsor** from the list of **Maintainers** or **Code-Owners** to help with the process.

* Fill in the proposal template.
  Be sure to include all required sections.
  Keep sections in the order prescribed in the template.

* Work with the **Sponsor** to identify the required reviewers and a timeline for review.

<!-- * Submit a pull request to the `enhancements` repository -->

* If discussion is needed the **Sponsor** can ask for a slot in the weekly Engineering Sync or schedule an ad-hoc meeting with the required reviewers.

* Iterate and incorporate feedback via the pull request.

* When review is complete The **Sponsor** will merge the request and update the status.

* **Sponsor** should assign an identifier.

* **Author** and **Sponsor** should add issues and/or PRs as needed to track implementation.

### When is a proposal required?

It is difficult to enumerate all the circumstances where a proposal would be required or not required.
Generally we will follow this process when making "substantial changes".
The definition of "substantial" is evolving and mainly determined by the core team and community.

When in doubt reach out to a **Maintainer** or **Code-Owner**.

**Generally speaking a proposal would not be required for**:

* Bug fixes that don't change advertised behavior

* Documentation fixes / updates

* Minor refactors within a single module

**Generally speaking proposals would be required for**:

* New features which add significant functionality

* Changes to existing features or code which require discussion

* Changes to public interfaces

* Responses to security related vulnerabilities found directly in the project code

* Changes to packaging and installation

* When a **Maintainer** or **Code-Owner** recommends that a change go through the proposal process

* Retroactively to capture current architecture, guideline, or process

### Minor Changes After Review

For minor changes or changes that are in the spirit of the review, updates can be made to the document without a new proposal.

*Example:* links to implementation

### Significant Changes After Review

For significant changes, a new proposal should be made and the original marked as replaced.

### Maintenance

TEPs should be reviewed for updates, replacements, or archiving on a regular basis.

### Sensitive Changes and Discussions

Certain types of changes need to be discussed and ratified before being made public due to timing of non-disclosed information.
In such (rare) cases, drafts and reviews will be conducted offline by **Authors**, **Code-Owners**, and **Maintainers** with the public proposals being updated when possible.

*Example:* when responding to undisclosed security vulnerabilities, we want to avoid inadvertently encouraging zero day attacks for deployed systems.

In such (rare) cases, we may make use of a private repo on a temporary basis to collect feedback before publishing to the public repo.

### Deferred to Implementation

* Definition of **Code-Owners** and **Maintainers**

* Whether or not to organize **TEP**s into sub directories for projects / areas

* Tooling around the creation / indexing of **TEP**s

* Making requirements required in addition to motivation

* Format recommendations for API surfaces / other formatted components.

* Decisions / guidelines on when a TEP is needed.

## Alternate Solutions

### Alt 1 Google Docs

**Pros:**

* Fits existing documents and templates used by many teams

**Cons:**

* Difficult to integrate with AI tools.

* Difficult to search and index

**Reason Rejected:**

* Want to standardize around a simple text format and use AI tools also for diagramming, etc.

## Background

With the rise of Agile software development practices and large open source projects, software development teams needed to devise new and lightweight (w.r.t to previous software architecture documents) ways of recording architecture proposals and decisions.
As Agile was born in part as a reaction to waterfall styles of planning and development and famously prioritized “Working software over comprehensive documentation”, so too there was a need to replace monolithic large software design specifications with something lighter weight but that still encouraged good architecture.

From this need for a new way of practicing software architecture a  body of work and theory has evolved around the concepts of “Architecture Decision Records” which in turn are also termed “Any Decision Record”, and RFCs or Enhancement proposals (PEP, KEP, REP).

In each case the core requirements of the process are that the team document the problem, the proposal / design, the status of the proposal, implications / follow on work, and any alternatives that were considered using a standard template and review process.

Just as in Agile planning, each team modifies the template and process to fit their needs.

### References

1. [Documenting Architecture Decisions (cognitect.com)](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)

2. [The most plagiarized Architecture Decision Record blog on the internet. | by Conall Daly | Medium](https://conalldalydev.medium.com/the-most-plagiarised-architecture-decision-record-blog-on-the-internet-c9dd2018c1d6)

3. [adr.github.io](https://adr.github.io/)

4. [When Should I Write an Architecture Decision Record \- Spotify Engineering : Spotify Engineering (atspotify.com)](https://engineering.atspotify.com/2020/04/when-should-i-write-an-architecture-decision-record/)

5. [Scaling Engineering Teams via RFCs: Writing Things Down \- The Pragmatic Engineer](https://blog.pragmaticengineer.com/scaling-engineering-teams-via-writing-things-down-rfcs/)

6. [Love Unrequited: The Story of Architecture, Agile, and How Architecture Decision Records Brought Them Together | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9801811)

7. [ray-project/enhancements: Tracking Ray Enhancement Proposals (github.com)](https://github.com/ray-project/enhancements)

8. [Kubernetes Enhancement Proposals](https://github.com/kubernetes/enhancements/blob/master/keps/sig-architecture/0000-kep-process/README.md)

9. [Dynamo Enhancement Proposals](https://github.com/ai-dynamo/enhancements/blob/main/README.md)

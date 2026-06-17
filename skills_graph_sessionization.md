Graph-Based Sessionization & Embedding Roadmap

Core Philosophy

The goal is NOT clustering.

The goal is NOT anomaly detection.

The goal is NOT Toponymy.

The goal is:

Learn high-quality session embeddings for security telemetry.

Everything else is downstream.

⸻

Problem Definition

Given arbitrary telemetry:

* Process logs
* Network logs
* Authentication logs
* EDR telemetry
* Cloud logs

Produce embeddings such that:

* Similar behavior is close together
* Different behavior is far apart
* Malicious activity is distinguishable from normal activity

The embedding is the primary research object.

⸻

Session Definition

A session is:

A connected group of related events representing a coherent activity episode.

The objective is not to perfectly reconstruct a user’s true session.

That is impossible from incomplete telemetry.

The objective is to construct meaningful activity groups that preserve behavioral relationships.

⸻

Why Graph Sessionization?

Traditional time-based sessionization assumes:

“Events close in time belong together.”

Graph sessionization assumes:

“Events connected through meaningful evidence belong together.”

This better matches the actual concept of an activity session.

⸻

Graph Sessionization Framework

Raw Events
↓
Extract Evidence Relationships
↓
Build Graph
↓
Connected Components
↓
Sessions

The graph is simply a mechanism for answering:

Which events belong together?

⸻

Graph Structure

Nodes

Version 1:

Node = Process Event

Each node stores metadata:

* event_id
* timestamp
* process_name
* hostname
* pid_hash
* arguments
* user
* other telemetry

Example:

powershell.exe PID=123

is a node.

⸻

Edges

An edge represents evidence that two events belong to the same activity episode.

Every edge must be defensible.

Each edge type should be implemented independently.

⸻

Edge Types

Parent-Child Edge

Strongest edge.

Represents process lineage.

Example:

cmd.exe
↓
powershell.exe

Implementation:

add_parent_child_edges()

Uses:

* parent_pid
* child_pid

Creates:

(parent, child)

Relationship type:

edge_type=“parent_child”

⸻

Rare Shared File Edge

Useful but requires filtering.

Good:

Both processes touched a rare file.

Bad:

Both processes touched:

* kernel32.dll
* explorer.exe
* common Windows artifacts

Implementation:

add_rare_file_edges()

Process:

1. Group by file
2. Calculate file frequency
3. Keep only rare files
4. Connect processes touching those files

Relationship type:

edge_type=“rare_file”

⸻

Rare Network Edge

Useful but requires filtering.

Good:

Multiple processes communicate with a rare destination.

Bad:

* Google
* Microsoft
* Windows Update

Implementation:

add_network_edges()

Process:

1. Group by destination
2. Calculate destination frequency
3. Keep rare destinations
4. Connect processes

Relationship type:

edge_type=“network”

⸻

Same User Edge

Potential future edge.

Currently lower priority because:

* weaker evidence
* missing values in ACME

Not required for Version 1.

⸻

GraphBuilder Design

NetworkX is the implementation layer.

The class owns the graph.

Example architecture:

class GraphBuilder:

def __init__(self):
    self.G = nx.Graph()
def add_process_nodes(self):
    pass
def add_parent_child_edges(self):
    pass
def add_rare_file_edges(self):
    pass
def add_network_edges(self):
    pass
def get_sessions(self):
    pass

The rest of the code should interact with GraphBuilder, not directly with NetworkX.

⸻

NetworkX Usage

Using NetworkX inside class methods is correct.

Examples:

self.G.add_node(…)
self.G.add_edge(…)
nx.connected_components(self.G)

NetworkX is the graph storage and graph algorithm library.

GraphBuilder contains the security logic.

⸻

Session Extraction

After all edges are added:

components = nx.connected_components(self.G)

Each connected component becomes a session.

Example:

A – B – C

D – E

F

Produces:

Session 1:
A B C

Session 2:
D E

Session 3:
F

This is the core sessionization algorithm.

⸻

Important Research Question

The research is NOT:

“Should I use a graph?”

The research IS:

Which edge types are valid evidence of related activity?

Every edge type must earn its place.

Future experiments:

Parent-child only
↓
Evaluate

Parent-child + rare file
↓
Evaluate

Parent-child + rare file + network
↓
Evaluate

Measure contribution of each evidence source.

⸻

Visualization Strategy

Do NOT visualize the full graph first.

Start with:

Stage 1

Toy graph on paper.

Stage 2

Tiny NetworkX graph.

Stage 3

Single connected component.

Stage 4

Color by edge type.

Stage 5

Export to Gephi.

⸻

First Metrics To Visualize

Most useful early visualization:

Connected component size distribution.

Questions:

Do we get:

* many small sessions
* some medium sessions
* few large sessions

or

* one giant component

or

* only singleton nodes

This immediately validates sessionization quality.

⸻

Embedding Roadmap

After sessionization is stable:

Phase 1:
Pooling baseline

Session
↓
Token embeddings
↓
Mean / Max pooling
↓
Session embedding

⸻

Phase 2:
GRU sequence model

Session
↓
GRU
↓
Embedding

Objective:

Predict next event.

⸻

Phase 3:
Transformer

Potential models:

* ModernBERT
* LogBERT
* Custom Transformer

Only if GRU saturates.

⸻

Future Graph Learning

Separate from sessionization graph.

Current graph:

Events
↓
Graph
↓
Sessions

Future graph:

Sessions
Users
Hosts
IPs
Domains

↓

GNN

Examples:

* GraphSAGE
* GAT
* GCN

Session embeddings become node features.

This is a later-stage research direction.

⸻

Evaluation Philosophy

Build the ruler first.

Metrics:

Security Metrics:

* Precision
* Recall
* Average Precision
* Recall@K

Geometry Metrics:

* Nearest-neighbor purity
* Cluster stability
* Silhouette score

Every future change must improve the ruler.

⸻

Key Principle

The graph is not the research contribution.

The edge definitions are.

NetworkX is not the research contribution.

The evidence model is.

The goal is to understand and own every edge type so the system becomes:

“My sessionization framework”

rather than

“Generated code I inherited.”
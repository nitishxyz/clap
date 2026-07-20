use std::collections::BTreeMap;

use crate::{Namespace, SlotId};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Entry {
    namespace: Namespace,
    slot: SlotId,
}

struct Edge {
    label: Vec<i32>,
    child: Node,
}

#[derive(Default)]
struct Node {
    children: BTreeMap<i32, Edge>,
    entries: Vec<Entry>,
}

/// Compressed exact-token radix index. Only terminal entries correspond to
/// materialized backend state; edge labels compact common token paths.
#[derive(Default)]
pub(crate) struct RadixIndex {
    root: Node,
}

impl RadixIndex {
    pub(crate) fn insert(&mut self, namespace: Namespace, tokens: &[i32], slot: SlotId) {
        Self::insert_at(&mut self.root, tokens, Entry { namespace, slot });
    }

    fn insert_at(node: &mut Node, tokens: &[i32], entry: Entry) {
        if tokens.is_empty() {
            match node.entries.binary_search(&entry) {
                Ok(_) => {}
                Err(index) => node.entries.insert(index, entry),
            }
            return;
        }

        let key = tokens[0];
        let Some(mut edge) = node.children.remove(&key) else {
            let mut child = Node::default();
            child.entries.push(entry);
            node.children.insert(
                key,
                Edge {
                    label: tokens.to_vec(),
                    child,
                },
            );
            return;
        };
        let shared = common_prefix(&edge.label, tokens);
        if shared == edge.label.len() {
            Self::insert_at(&mut edge.child, &tokens[shared..], entry);
            node.children.insert(key, edge);
            return;
        }

        let old_suffix = edge.label.split_off(shared);
        let common_label = edge.label;
        let mut branch = Node::default();
        branch.children.insert(
            old_suffix[0],
            Edge {
                label: old_suffix,
                child: edge.child,
            },
        );
        Self::insert_at(&mut branch, &tokens[shared..], entry);
        node.children.insert(
            key,
            Edge {
                label: common_label,
                child: branch,
            },
        );
    }

    pub(crate) fn remove(&mut self, namespace: Namespace, tokens: &[i32], slot: SlotId) {
        Self::remove_from(&mut self.root, tokens, Entry { namespace, slot });
    }

    fn remove_from(node: &mut Node, tokens: &[i32], entry: Entry) {
        if tokens.is_empty() {
            if let Ok(index) = node.entries.binary_search(&entry) {
                node.entries.remove(index);
            }
            return;
        }
        let key = tokens[0];
        let Some(mut edge) = node.children.remove(&key) else {
            return;
        };
        if tokens.starts_with(&edge.label) {
            Self::remove_from(&mut edge.child, &tokens[edge.label.len()..], entry);
        }
        if edge.child.entries.is_empty() && edge.child.children.is_empty() {
            return;
        }
        if edge.child.entries.is_empty() && edge.child.children.len() == 1 {
            let (_, next) = edge.child.children.pop_first().expect("one child");
            edge.label.extend(next.label);
            edge.child = next.child;
        }
        node.children.insert(key, edge);
    }

    pub(crate) fn node_count(&self) -> usize {
        fn count(node: &Node) -> usize {
            node.children
                .values()
                .map(|edge| 1 + count(&edge.child))
                .sum()
        }
        count(&self.root)
    }

    pub(crate) fn clear(&mut self) {
        self.root = Node::default();
    }
}

fn common_prefix(left: &[i32], right: &[i32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

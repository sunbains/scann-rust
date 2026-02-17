//! Document ID management.
//!
//! This module provides types for associating document IDs with datapoints.

use crate::types::DatapointIndex;
use std::collections::HashMap;

/// A document ID, which can be either a string or an integer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DocId {
    /// String document ID.
    String(String),
    /// Integer document ID.
    Int(i64),
}

impl DocId {
    /// Create a string document ID.
    pub fn string(s: impl Into<String>) -> Self {
        DocId::String(s.into())
    }

    /// Create an integer document ID.
    pub fn int(i: i64) -> Self {
        DocId::Int(i)
    }

    /// Get the string value if this is a string ID.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            DocId::String(s) => Some(s),
            DocId::Int(_) => None,
        }
    }

    /// Get the integer value if this is an integer ID.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            DocId::String(_) => None,
            DocId::Int(i) => Some(*i),
        }
    }
}

impl From<String> for DocId {
    fn from(s: String) -> Self {
        DocId::String(s)
    }
}

impl From<&str> for DocId {
    fn from(s: &str) -> Self {
        DocId::String(s.to_string())
    }
}

impl From<i64> for DocId {
    fn from(i: i64) -> Self {
        DocId::Int(i)
    }
}

impl From<u64> for DocId {
    fn from(i: u64) -> Self {
        DocId::Int(i as i64)
    }
}

impl From<u32> for DocId {
    fn from(i: u32) -> Self {
        DocId::Int(i as i64)
    }
}

impl std::fmt::Display for DocId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocId::String(s) => write!(f, "{}", s),
            DocId::Int(i) => write!(f, "{}", i),
        }
    }
}

/// Collection of document IDs with lookup support.
#[derive(Debug, Clone, Default)]
pub struct DocIdCollection {
    /// Document IDs in order.
    docids: Vec<DocId>,

    /// Reverse lookup from DocId to index.
    lookup: HashMap<DocId, DatapointIndex>,
}

impl DocIdCollection {
    /// Create an empty collection.
    pub fn new() -> Self {
        Self {
            docids: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    /// Create a collection with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            docids: Vec::with_capacity(capacity),
            lookup: HashMap::with_capacity(capacity),
        }
    }

    /// Add a document ID.
    pub fn push(&mut self, docid: DocId) {
        let index = self.docids.len() as DatapointIndex;
        self.lookup.insert(docid.clone(), index);
        self.docids.push(docid);
    }

    /// Get the number of document IDs.
    pub fn len(&self) -> usize {
        self.docids.len()
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.docids.is_empty()
    }

    /// Get a document ID by index.
    pub fn get(&self, index: DatapointIndex) -> Option<&DocId> {
        self.docids.get(index as usize)
    }

    /// Look up the index for a document ID.
    pub fn lookup(&self, docid: &DocId) -> Option<DatapointIndex> {
        self.lookup.get(docid).copied()
    }

    /// Check if a document ID exists.
    pub fn contains(&self, docid: &DocId) -> bool {
        self.lookup.contains_key(docid)
    }

    /// Iterate over document IDs.
    pub fn iter(&self) -> impl Iterator<Item = &DocId> {
        self.docids.iter()
    }

    /// Clear the collection.
    pub fn clear(&mut self) {
        self.docids.clear();
        self.lookup.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docid_string() {
        let docid = DocId::string("test_doc");
        assert_eq!(docid.as_string(), Some("test_doc"));
        assert_eq!(docid.as_int(), None);
    }

    #[test]
    fn test_docid_int() {
        let docid = DocId::int(42);
        assert_eq!(docid.as_int(), Some(42));
        assert_eq!(docid.as_string(), None);
    }

    #[test]
    fn test_docid_collection() {
        let mut collection = DocIdCollection::new();
        collection.push(DocId::string("doc1"));
        collection.push(DocId::string("doc2"));
        collection.push(DocId::int(123));

        assert_eq!(collection.len(), 3);
        assert_eq!(collection.get(0), Some(&DocId::string("doc1")));
        assert_eq!(collection.lookup(&DocId::string("doc2")), Some(1));
        assert_eq!(collection.lookup(&DocId::int(123)), Some(2));
        assert!(collection.contains(&DocId::string("doc1")));
        assert!(!collection.contains(&DocId::string("missing")));
    }
}

# nodes/tests/test_add_nodes.py
# Tests for adding nodes to the network.

import pytest

from nodes.builder import NetworkBuilder
from nodes.network.single_nodes import Token
from nodes.enums import *
from nodes.network import Ref_Token

# Import the symProps from sim.py
from .sims.sim import symProps

import torch

@pytest.fixture
def network():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_add_po_token_to_driver(network):
    """Test adding a PO token to the driver tensor."""
    # Create a new PO token
    token = Token(type=Type.PO, features={
        TF.SET: Set.DRIVER,
        TF.ANALOG: 0,
        TF.PRED: True
    })
    
    # Add the token to the driver tensor
    token_reference: Ref_Token = network.node.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.DRIVER
    
    # Get the token from the tensor
    token = network.sets[Set.DRIVER].token_op.get_single_token(token_reference)
    
    
    # Verify the token properties
    assert token.tensor[TF.TYPE] == Type.PO
    assert token.tensor[TF.SET] == Set.DRIVER
    assert token.tensor[TF.ANALOG] == 0
    assert token.tensor[TF.PRED] == B.TRUE
    assert token.tensor[TF.INHIBITOR_THRESHOLD] == 110  # Default for PO tokens

def test_add_rb_token_to_recipient(network):
    """Test adding a RB token to the recipient tensor."""
    # Create a new RB token
    token = Token(type=Type.RB, features={
        TF.SET: Set.RECIPIENT,
        TF.ANALOG: 0
    })
    
    # Add the token to the recipient tensor
    token_reference: Ref_Token = network.node.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.RECIPIENT
    
    # Get the token from the tensor
    token = network.sets[Set.RECIPIENT].token_op.get_single_token(token_reference)
    
    # Verify the token properties
    assert token[TF.TYPE] == Type.RB
    assert token[TF.SET] == Set.RECIPIENT
    assert token[TF.ANALOG] == 0
    assert token[TF.INHIBITOR_THRESHOLD] == 220  # Default for RB tokens

def test_add_p_token_to_memory(network):
    """Test adding a P token to the memory tensor."""
    # Create a new P token
    token = Token(type=Type.P, features={
        TF.SET: Set.MEMORY,
        TF.ANALOG: 0
    })
    
    # Add the token to the memory tensor
    token_reference: Ref_Token = network.node.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.MEMORY
    
    # Get the token from the tensor
    token = network.sets[Set.MEMORY].token_op.get_single_token(token_reference)
    
    # Verify the token properties
    assert token[TF.TYPE] == Type.P
    assert token[TF.SET] == Set.MEMORY
    assert token[TF.ANALOG] == 0
    assert token[TF.INHIBITOR_THRESHOLD] == 440  # Default for P tokens

def test_add_multiple_tokens_new_set(network):
    """Test adding multiple tokens to the new_set tensor."""
    # Create multiple tokens
    tokens = [
        Token(type=Type.PO, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0, TF.PRED: True}),
        Token(type=Type.RB, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0}),
        Token(type=Type.P, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0})
    ]
    
    network[Set.NEW_SET].print()
    # Add the tokens to the new_set tensor
    token_references: list[Ref_Token] = [network.node.add_token(token) for token in tokens]
    
    # Verify all tokens were added correctly
    assert len(token_references) == 3
    assert all(token_reference.set == Set.NEW_SET for token_reference in token_references)
    
    # Verify the tokens have different IDs
    assert len(set(token_reference.ID for token_reference in token_references)) == 3

def test_add_token_with_custom_features(network):
    """Test adding a token with custom features to the driver tensor."""
    # Create a new PO token with custom features
    token = Token(type=Type.PO, features={
        TF.SET: Set.DRIVER,
        TF.ANALOG: 1,
        TF.PRED: True,
        TF.ACT: 0.5,
        TF.MAX_ACT: 1.0,
        TF.NET_INPUT: 0.3
    })
    
    # Add the token to the driver tensor
    token_reference: Ref_Token = network.node.add_token(token)
    
    network.sets[Set.DRIVER].print(f_types=[TF.ID, TF.TYPE, TF.SET, TF.ANALOG, TF.PRED, TF.ACT, TF.MAX_ACT, TF.NET_INPUT, TF.INHIBITOR_THRESHOLD])
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.DRIVER
    
    # Get the token from the tensor
    token = network.sets[Set.DRIVER].token_op.get_single_token(token_reference)
    
    # Verify the token properties
    assert token[TF.TYPE] == Type.PO
    assert token[TF.SET] == Set.DRIVER
    assert token[TF.ANALOG] == 1
    assert token[TF.PRED] == B.TRUE
    assert token[TF.ACT] == 0.5
    assert token[TF.MAX_ACT] == 1.0
    assert token[TF.NET_INPUT] == 0.3
    assert token[TF.INHIBITOR_THRESHOLD] == 110  # Default for PO tokens

def test_add_token_to_full_tensor(network):
    """Test adding a token when the tensor is full."""
    # First, fill up the new_set tensor
    initial_count = network[Set.NEW_SET].nodes.size(dim=0)

    new_count = int(initial_count * network[Set.NEW_SET].expansion_factor) 
    if new_count < 5:                                                   # minimum expansion is 5
            new_count = 5

    # Add tokens until the tensor is full
    references: list[Ref_Token] = []
    tokens: list[Token] = []
    for i in range(initial_count):
        token = Token(type=Type.PO, features={
            TF.SET: Set.NEW_SET,
            TF.ANALOG: 0,
            TF.PRED: True
        })
        reference: Ref_Token = network.node.add_token(token)
        references.append(reference)
        tokens.append(network.token_op.get_single_token(reference))
    
    # Now add one more token, which should trigger tensor expansion
    token = Token(type=Type.PO, features={
        TF.SET: Set.NEW_SET,
        TF.ANALOG: 0,
        TF.PRED: True
    })
    token_reference: Ref_Token = network.node.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.NEW_SET
    
    network[Set.NEW_SET].print()
    # Verify the tensor was expanded
    assert network[Set.NEW_SET].nodes.size(dim=0) > initial_count
    assert network[Set.NEW_SET].nodes.size(dim=0) == new_count
    print(network[Set.NEW_SET].nodes.size(dim=0))
    print(network[Set.NEW_SET].links[Set.NEW_SET].size(dim=0), network[Set.NEW_SET].links[Set.NEW_SET])
    assert network[Set.NEW_SET].nodes.size(dim=0) == network[Set.NEW_SET].links[Set.NEW_SET].size(dim=0)

    # Verify references still give the same tokens
    for reference, token in zip(references, tokens):
        new_token = network.token_op.get_single_token(reference)
        assert torch.equal(new_token.tensor, token.tensor)
        assert new_token.ID == token.ID
        assert new_token.set == token.set

def test_type_after_expansion(network):
    """Test that the type of a nodes tensor is the same after expansion."""
    network.sets[Set.DRIVER].print()
    assert type(network.sets[Set.DRIVER].nodes[0, TF.ACT].item()) == float
    network.sets[Set.DRIVER].expand_tensor()
    network.sets[Set.DRIVER].print()
    assert type(network.sets[Set.DRIVER].nodes[0, TF.ACT].item()) == float

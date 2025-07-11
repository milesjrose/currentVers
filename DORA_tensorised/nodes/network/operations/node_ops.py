# nodes/network/operations/node_ops.py
# Node operations for Network class

from ...enums import *
from ..single_nodes import Token, Semantic, Ref_Token, Ref_Semantic

class NodeOperations:
    """
    Node operations for the Network class.
    Handles node management.
    """
    
    def __init__(self, network):
        """
        Initialize NodeOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network = network
        self.sets = network.sets

    def get_pmode(self):                                                    # Get p_mode in driver and recipient
        """
        Get parent mode of P units in driver and recipient. Used in time steps activations.
        (driver, recipient)
        """
        self.sets[Set.DRIVER].p_get_mode()
        self.sets[Set.RECIPIENT].p_get_mode()
    
    def initialise_p_mode(self, set: Set = Set.RECIPIENT):                  # Initialise p_mode in the given set
        """
        Initialise p_mode in the given set.
        (default: recipient)

        Args:
            set (Set, optional): The set to initialise p_mode in. (Defaults to recipient)
        """
        self.sets[set].initialise_p_mode()
    
    def get_weight_lengths(self):                                           # get weight lenghts in active memory
        """
        Get weight lengths of PO units. Used in run initialisation.
        (driver, recipient, memory, new_set)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.MEMORY, Set.NEW_SET]
        for set in sets:
            self.sets[set].po_get_weight_length()
    
    def add_token(self, token: Token):                                      # Add a token to the given set
        """
        Add a token to the network.
        - Added to the set specified in the token.

        Args:
            token (network.Token): The token to add.
        
        Returns:
            network.Ref_Token: A reference to the token.
        
        Raises:
            ValueError: If the token set feature is not a valid set.
        """
        add_set = int(token.tensor[TF.SET])
        if add_set not in [set.value for set in Set]:
            raise ValueError("Invalid set in token feature.")
        
        reference = self.sets[add_set].add_token(token)
        return reference

    def del_token(self, ref_token: Ref_Token):                              # Delete a token
        """
        Delete a referenced token from the network.

        Args:
            ref_token (network.Ref_Token): A reference to the token to delete.
        """
        self.sets[ref_token.set].del_token(ref_token)

    def add_semantic(self, semantic: Semantic):                             # Add a semantic
        """
        Add a semantic to the given set.

        Args:
            semantic (network.Semantic): The semantic to add.
        
        Raises:
            ValueError: If provided semantic is not semantic type.
        """
        if semantic.tensor[SF.TYPE] != Type.SEMANTIC:
            raise ValueError("Cannot add non-semantic to semantic set.")
        
        self.semantics.add_semantic(semantic)

    def del_semantic(self, ref_semantic: Ref_Semantic):                     # Delete a semantic
        """
        Delete a semantic from the semantics.
        
        Args:
            ref_semantic (network.Ref_Semantic): A reference to the semantic to delete.

        Raises:
            ValueError: If ref_semantic is not provided.
        """
        self.semantics.del_semantic(ref_semantic)

    def set_sem_max_input(self):                                            # Set the maximum input for the semantics
        """
        Set the maximum input for the semantics.
        """
        max_input = self.semantics.get_max_input()
        self.semantics.set_max_input(max_input)
    
    def get_value(self, reference, feature):                                # Get the value of a feature for a referenced token or semantic
        """
        Get the value of a feature for a referenced token or semantic.

        Args:
            reference (Ref_Token or Ref_Semantic): A reference to the token or semantic to get the value of.
            feature (TF or SF): The feature to get the value of.

        Returns:
            float: The value of the feature.

        Raises:
            ValueError: If the reference is not a token or semantic. Or feature type and reference type mismatch.
        """
        if isinstance(reference, Ref_Token):
            if isinstance(feature, TF):
                return self.sets[reference.set].token_op.get_feature(reference, feature)
            else:
                raise ValueError("Referenced a token, but feature is not a token feature.")
        elif isinstance(reference, Ref_Semantic):
            if isinstance(feature, SF):
                return self.semantics.get_feature(reference, feature)
            else:
                raise ValueError("Referenced a semantic, but feature is not a semantic feature.")
        else:
            raise ValueError("Invalid reference type.")
    
    def set_value(self, reference, feature, value):                         # Set the value of a feature for a referenced token or semantic
        """
        Set the value of a feature for a referenced token or semantic.

        Args:
            reference (Ref_Token or Ref_Semantic): A reference to the token or semantic to set the value of.
            feature (TF or SF): The feature to set the value of.
            value (float or Enum): The value to set the feature to.

        Raises:
            ValueError: If the reference is not a token or semantic. Or feature type and reference type mismatch.
        """
        if isinstance(reference, Ref_Token):
            if isinstance(feature, TF):
                self.sets[reference.set].token_op.set_feature(reference, feature, value)
            else:
                raise ValueError("Referenced a token, but feature is not a token feature.")
        elif isinstance(reference, Ref_Semantic):
            if isinstance(feature, SF):
                self.semantics.set_feature(reference, feature, value)
            else:
                raise ValueError("Referenced a semantic, but feature is not a semantic feature.")
        else:
            raise ValueError("Invalid reference type.")
     
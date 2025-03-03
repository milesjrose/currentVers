# nodesMemTypes.py 
# Classes for segments of memory, and set-specific tensor operations
import torch
from nodeEnums import *
import tensorOps as tOps


class TokenTensor(object):
    def __init__(self, floatTensor, boolTensor, connections, links):
        self.nodes: torch.Tensor = floatTensor
        self.nodesBool: torch.Tensor = boolTensor
        self.masks = self.cache_masks()
        self.indicies = self.cache_masks()
        self.analogs, self.analog_counts = self.analog_node_count()

        # Weighted undirected adj matrix (NxS), connections between set tokens and semantics
        self.links = links()
        # Unweighted directed connections between token of same set. Connetion is from parent to child (i.e [i, j] = 1 means “node i is the parent of node j”)
        self.connections = connections

    def cache_masks(self, types_to_recompute = None):   # Compute and cache masks, specify types to recompute via list of tokenTypes
        if types_to_recompute == None:                      #  If no type specified, recompute all
            types_to_recompute = [Type.PO, Type.RB, Type.P, Type.GROUP]

        masks = []
        for token_type in [Type.PO, Type.RB, Type.P, Type.GROUP]:
            if token_type in types_to_recompute:
                masks.append(self.computeMasks(token_type)) # Recompute mask
            else:
                masks.append(self.masks[token_type])        # Use cached mask

        self.masks: torch.Tensor = torch.stack(masks, dim=0)
    
    def compute_mask(self, token_type: Type):           # Compute the mask for a token type
        mask = (self.nodes[:, TF.TYPE] == token_type) 
        return mask
    
    def get_mask(self, token_type: Type):               # Returns mask for given token type
        return self.masks[token_type]                   

    def get_combined_mask(self, n_types: list[Type]):   # Returns combined mask of give types
        masks = [self.masks[i] for i in n_types]
        return torch.logical_or.reduce(masks)

    def add_nodes(self, nodes):                                     # TODO: Add nodes 
        # As tensor is non-extensible, keep some headroom. If the tensor is full create new tensor with ratio 1.1, o.w just add node
        # TODO: allow for adding nodes - tensor non-extensible, so must create new tensor then add node
        return None
    
    def del_Nodes(self, nodes):                                     # TODO: Delete nodes
        # Tensor size is static, so just set the nodes deleted to zero - then recompute masks to exclude it.
        # TODO: When deleted nodes ratio hits threshold, compress the tensor to remove all deleted nodes.
        # TODO: Impletment
        return None
    
    def analog_node_count(self):                                    # Updates list of analogs in tensor, and their node counts
        self.analogs, self.analog_counts = torch.unique(self.nodes[:, TF.ANALOG], return_counts=True)

    # =====================[ TOKEN FUNCTIONS ]=======================
    def initialise_float(self, n_type: list[Type], features: list[TF]): # Initialise given features
        type_mask = self.get_combined_mask(n_type)                      # Get mask of nodes to update
        init_subt = self.nodes[type_mask, features]                     # Get subtensor of features to intialise
        self.nodes[type_mask, features] = torch.zeros_like(init_subt)   # Set features to 0
    
    def initialise_input(self, n_type: list[Type], refresh):        # Initialize inputs to 0, and td_input to refresh.
        type_mask = self.get_combined_mask(n_type)
        self.nodes[type_mask, TF.TD_INPUT] = refresh                    # Set td_input to refresh
        features = [TF.BU_INPUT,TF.LATERAL_INPUT,TF.MAP_INPUT,TF.NET_INPUT]
        self.initialise_float(n_type, features)                         # Set types to 0.0

    def initialise_act(self, n_type: list[Type]):                   # Initialize act to 0.0,  and call initialise_inputs
        self.initialise_input(n_type, 0.0)
        self.initialise_float(n_type, [TF.ACT])

    def initialise_state(self, n_type: list[Type]):                 # Set self.retrieved to false, and call initialise_act
        self.initialise_act(n_type)
        self.initialise_float(n_type, [TF.RETRIEVED])                       
        
    def update_act(self, gamma, delta, HebbBias):                   # Update act of nodes
        net_input_types = [
            TF.TD_INPUT,
            TF.BU_INPUT,
            TF.LATERAL_INPUT
        ]
        net_input = self.nodes[:, net_input_types].sum(dim=1, keepdim=True) # sum non mapping inputs
        net_input += self.nodes[:, self.map_input] * HebbBias               # Add biased mapping input
        acts = self.nodes[:, TF.ACT]                                        # Get node acts
        delta_act = gamma * net_input * (1.1 - acts) - (delta * acts)       # Find change in act for each node
        acts += delta_act                                                   # Update acts
        
        self.nodes[(self.nodes[:, TF.ACT] > 1.0), TF.ACT] = 1.0             # Limit activation to 1.0 or below
        self.nodes[(self.nodes[:, TF.ACT] < 0.0), TF.ACT] = 0.0             # Limit activation to 0.0 or above                                      # update act

    def zero_lateral_input(self, n_type: list[Type]):               # Set lateral_input to 0 (to allow synchrony at different levels by 0-ing lateral inhibition at that level (e.g., to bind via synchrony, 0 lateral inhibition in POs).
        self.initialise_float(n_type, [TF.LATERAL_INPUT])
    
    def update_inhibitor_input(self, n_type: list[Type]):           # Update inputs to inhibitors by current activation for nodes of type n_type
        mask = self.get_combined_mask(n_type)
        self.nodes[mask, TF.INHIBITOR_INPUT] += self.nodes[mask, TF.ACT]

    def reset_inhibitor(self, n_type: list[Type]):                  # Reset the inhibitor input and act to 0.0 for given type
        mask = self.get_combined_mask(n_type)
        self.nodes[mask, TF.INHIBITOR_INPUT] = 0.0
        self.nodes[mask, TF.INHIBITOR_ACT] = 0.0
    
    def update_inhibitor_act(self, n_type: list[Type]):             # Update the inhibitor act for given type
        type_mask = self.get_combined_mask(n_type)
        input = self.nodes[type_mask, TF.INHIBITOR_INPUT]
        threshold = self.nodes[type_mask, TF.INHIBITOR_THRESHOLD]
        nodes_to_update = (input >= threshold)                      # if inhib_input >= inhib_threshold
        self.nodes[nodes_to_update, TF.INHIBITOR_ACT] = 1.0         # then set to 1

    # =======================[ P FUNCTIONS ]========================
    def p_initialise_mode(self):                                    # Initialize all p.mode back to neutral.
        p = self.get_mask(Type.P)
        self.nodes[p, TF.MODE] = Mode.NEUTRAL

    def p_get_mode(self):                                           # Set mode for all P units
        # Pmode = Parent: child RB act> parent RB act / Child: parent RB act > child RB act / Neutral: o.w
        p = self.get_mask(Type.P)
        rb = self.get_mask(Type.RB)
        child_input = torch.matmul(                                 # Px1 matrix: sum of child rb for each p
            self.connections[p, rb],
            self.nodes[rb, TF.ACT]
        )
        parent_input = torch.matmult(                               # Px1 matrix: sum of parent rb for each p
            torch.transpose(self.connections)[p, rb],
            self.nodes[rb]
        )
        # Get global masks of p, by mode
        input_diff = parent_input - child_input                     # (input_diff > 0) <-> (parents > childs)
        child_p = tOps.sub_union(p, (input_diff[:, 0] > 0.0))       # (input_diff > 0) -> (parents > childs) -> (p mode = child)
        parent_p = tOps.sub_union(p, (input_diff[:, 0] < 0.0))      # (input_diff < 0) -> (parents < childs) -> (p mode = parent) 
        neutral_p = tOps.sub_union(p, (input_diff[:, 0] == 0.0))    # input_diff == 0 -> p mode = neutral
        # Set mode values:
        self.nodes[child_p, TF.MODE] = Mode.CHILD                   
        self.nodes[parent_p, TF.MODE] = Mode.PARENT
        self.nodes[neutral_p, TF.MODE] = Mode.NEUTRAL

    # =======================[ RB FUNCTIONS ]========================
    def rb_initiaise_times_fired(self):                             # Initialise all RBs times fired NOTE: Never used?
        self.initialise_float(Type.RB, TF.TIMES_FIRED)

    def rb_update_times_fired(self):                                # TODO: Implement   NOTE: ALso never used?
        pass

    # =======================[ PO FUNCTIONS ]========================
    def po_get_weight_length(self, links):                          # Sum value of links with weight > 0.1 for all PO nodes
        po = self.get_mask(Type.PO)                                 # mask links with PO
        mask = self.links[po] > 0.1                                 # Create sub mask for links with weight > 0.1
        weights = (self.links[po] * mask).sum(dim=1, keepdim = True) # Sum links > 0.1
        self.nodes[po, TF.SEM_COUNT] = weights                      # Set semNormalisation

    def po_get_max_semantic_weight(self):                           # Get max link weight for all PO nodes
        po = self.get_mask(Type.PO)
        max_values, _ = torch.max(self.links[po], dim=1, keepdim=True)# (max_values, _) unpacks tuple returned by torch.max
        self.nodes[po, TF.MAX_SEM_WEIGHT] = max_values              # Set max


class DriverTensor(TokenTensor):
    def __init__(self, floatTensor, boolTensor, connections):   
        super().__init__(floatTensor, boolTensor, connections)
    
    def check_local_inhibitor(self):                                # Return true if any PO.inhibitor_act == 1.0
        po = self.get_mask(Type.PO)
        return torch.any(self.nodes[po, TF.INHIBITOR_ACT] == 1.0) 

    def check_global_inhibitor(self):                               # Return true if any RB.inhibitor_act == 1.0
        rb = self.get_mask(Type.RB)
        return torch.any(self.nodes[rb, TF.INHIBITOR_ACT] == 1.0) 
    
    # ==============[ DRIVER UPDATE INPUT FUNCTIONS ]===============
    def update_input(self, as_DORA):                                # Update all input in driver
        self.update_input_p_parent()
        self.update_input_p_child(as_DORA)
        self.update_input_rb(as_DORA)
        self.update_input_po(as_DORA)

    def update_input_p_parent(self):                                # P units in parent mode
        # Exitatory: td (my Groups) / bu (my RBs)
        # Inhibitory: lateral (other P units in parent mode*3), inhibitor.
        # 1). get masks
        p = self.get_mask(Type.P)                                   # Boolean mask for P nodes
        p = tOps.refine_mask(self.nodes, p, TF.MODE, Mode.PARENT)   # Boolean mask for Parent P nodes
        group = self.get_mask(Type.GROUP)                           # Boolean mask for GROUP nodes
        rb = self.get_mask(Type.RB)                                 # Boolean mask for RB nodes

        # Exitatory input:
        # 2). TD_INPUT: my_groups
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, group],                             # Masks connections between p[i] and its groups
            self.nodes[group, TF.ACT]                               # each p node -> sum of act of connected group nodes
            )
        # 3). BU_INPUT: my_RBs
        self.nodes[p, TF.BU_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, rb],                                # Masks connections between p[i] and its rbs
            self.nodes[rb, TF.ACT]                                  # Each p node -> sum of act of connected rb nodes
            )  
        
        # Inhibitory input:
        # 4). LATERAL_INPUT: (3 * other parent p nodes in driver), inhibitor
        # 4a). Create tensor mask of parent p nodes, and a tensor to connect p nodes to each other
        diag_zeroes = tOps.diag_zeros(sum(p))                       # adj matrix connection connecting parent ps to all but themselves
        # 4b). 3 * other parent p nodes in driver
        self.nodes[p, TF.LATERAL_INPUT] -= torch.mul(3, torch.matmul(
            diag_zeroes,                                            # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
            self.nodes[p, TF.ACT]                                   # Each parent p node -> 3*(sum of all other parent p nodes)
        ))

    def update_input_p_child(self, as_DORA):                        # P units in child mode:
        # Exitatory: td (my parent RBs), (if phase_set>1: my groups)
        # Inhibitory: lateral (Other p in child mode), (if DORA_mode: PO acts / Else: POs not connected to same RBs)
        # 1). get masks
        p = self.get_mask(Type.P)
        p = tOps.refine_mask(self.nodes, p, TF.MODE, Mode.CHILD)    # Boolean mask for Child P nodes
        group = self.get_mask(Type.GROUP)                           # Boolean mask for GROUP nodes
        rb = self.get_mask(Type.RB)                                 # Boolean mask for RB nodes
        po = self.get_mask(Type.PO)

        # Exitatory input:
        # 2). TD_INPUT: my_groups and my_parent_RBs
        # 2a). groups
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, group],                             # Masks connections between p[i] and its groups
            self.nodes[group, TF.ACT]                               # For each p node -> sum of act of connected group nodes
            )
        # 2b). parent_rbs
        t_con = torch.transpose(self.connections)                   # transpose, so gives child -> parent connections
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs matrix (sum(p) x 1) of values to add to current input value
            t_con[p, rb],                                           # Masks connections between p[i] and its rbs
            self.nodes[rb, TF.ACT]                                  # For each p node -> sum of act of connected parent rb nodes
            )
        
        # Inhibitory input:
        # 3). LATERAL_INPUT: (Other p in child mode), (if DORA_mode: PO acts / Else: POs not connected to same RBs)
        # 3a). other p in child mode
        diag_zeroes = tOps.diag_zeros(sum(p))                       # adj matrix connection connecting child ps to all but themselves
        self.nodes[p, TF.LATERAL_INPUT] -= torch.matmul(
            diag_zeroes,                                            # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
            self.nodes[p, TF.ACT]                                   # Each child p node -> 3*(sum of all other parent p nodes)
        )
        # 3b). if as_DORA: Objects not connected to same RBs
        if as_DORA:
            obj = tOps.refine_mask(self.nodes, po, TF.PRED, B.FALSE)# Boolean mask for objects
            ones = torch.ones((sum(p), sum(obj)))                   # tensor connecting every p to every object
            self.nodes[p, TF.LATERAL_INPUT] -= torch.matmul(    
                ones,                                               # connects all p to all object
                self.nodes[obj, TF.ACT]                             # Each  p node -> sum of all object acts
            )
        # 3c). Else: Object acts
        else: 
            # 3ci). Create masks
            obj = self.get_mask(Type.PO)                            # get PO mask
            obj = tOps.refine_mask(self.nodes, po, TF.PRED, B.TRUE) # refine mask for objects only
            # 3cii). Find objects not connected to me               # NOTE: check if p.myRB contains p.myParentRBs !not true! TODO: fix
            ud_con = torch.bitwise_or(self.connections, t_con)      # undirected connections tensor (OR connections with its transpose)
            shared = torch.matmul(ud_con[p, rb], ud_con[rb, obj])   # PxObject tensor, shared[i][j] > 1 if p[i] and object[j] share an RB, 0 o.w
            shared = torch.gt(shared, 0).int()                      # now shared[i][j] = 1 if p[i] and object[j] share an RB, 0 o.w
            non_shared = 1 - shared                                 # non_shared[i][j] = 0 if p[i] and object[j] share an RB, 1 o.w
            # 3cii). update input using non shared objects
            self.nodes[p, TF.LATERAL_INPUT] -= torch.matmul(
                non_shared,                                         # sum(p)xsum(object) matrix, listing non shared objects for each p
                self.nodes[obj, TF.ACT]                             # sum(objects)x1 matrix, listing act of each object
            )
   
    def update_input_rb(self, as_DORA):                             # TODO: implement
        pass
    
    def update_input_po(self, as_DORA):                             # TODO: implement
        pass
    # --------------------------------------------------------------


class RecipientTensor(TokenTensor):
    def __init__(self, floatTensor, boolTensor, connections):
        super().__init__(floatTensor, boolTensor, connections)

    # ============[ RECIPIENT UPDATE INPUT FUNCTIONS ]==============
    def update_input(self, as_DORA, phase_set, lateral_input_level, ignore_object_semantics=False): # Update all input in recipient
        self.update_input_p_parent()
        self.update_input_p_child(as_DORA)
        self.update_input_rb(as_DORA)
        self.update_input_po(as_DORA)

    def update_input_p_parent(self):                                # TODO: implement
        pass

    def update_input_p_child(self):                                 # TODO: implement
        pass
    
    def update_input_rb(self, as_DORA, phase_set, lateral_input_level): # TODO: implement
        pass
    
    def update_input_po(self, as_DORA, phase_set, lateral_input_level, ignore_object_semantics=False): # TODO: implement
        pass
    # --------------------------------------------------------------
    
class SemanticTensor(TokenTensor):                                  # TODO: implement
    def __init__(self):
        pass
    
    def initialise_sem():                                           # TODO: implement
        pass

class Links(object):    # Weighted connections between nodes
    def __init__(self, driverLinks, recipientLinks, semLinks):  # Takes weighted adjacency matrices
        self: torch.Tensor = driverLinks
        self.recipient: torch.Tensor = recipientLinks
        self.semantics: torch.Tensor = semLinks
    
    def add_links(self, set: Set, links):                           # TODO: implement
        pass


class Mappings(object): # 3D tensor storing mapping and hypothesis information
    def __init__(self, connections):    # Takes 3D tensor, of stacked 2D adjacency matrices
        # Takes 3D tensor, of stacked 2D adjacency matrices
        self.adj_matrix: torch.Tensor = connections
    
    def weights(self):
        return self.adj_matrix[:, :, MappingFields.WEIGHT]
    
    def hypotheses(self):
        return self.adj_matrix[:, :, MappingFields.HYPOTHESIS]
    
    def max_hyps(self):
        return self.adj_matrix[:, :, MappingFields.MAX_HYP]
    
    def updateHypotheses(self, hypotheses):                         # TODO: implement
        pass
    
    def add_mappings(self,  mappings):                              # TODO: implement
        pass
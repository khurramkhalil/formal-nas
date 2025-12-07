"""
Synthesis Baselines for Comparative Evaluation.

Implements Random Search (Rejection Sampling) to provide a fair baseline
comparing "SMT-Based Synthesis" vs "Guess-and-Check".
"""

import random
import time
from typing import List, Dict, Any, Optional

class RandomSearchBaseline:
    """
    Attempts to find valid architectures by randomly sampling operations and connectivity.
    """
    def __init__(self, max_nodes: int = 8, input_res: int = 32):
        self.max_nodes = max_nodes
        self.input_res = input_res
        
    def generate_random_props(self) -> List[Dict[str, Any]]:
        """Generate a random set of node properties (unverified)."""
        props = []
        for i in range(1, self.max_nodes):
            # Sample Ops: 1=Conv, 2=Pool, 3=Add, 4=Concat
            # Weighted distinct-ness? No, pure uniform random.
            op = random.choice([1, 2, 3, 4]) 
            
            p = {
                'id': i,
                'op': op,
                'in1': random.randint(0, i-1),
                'in2': random.randint(0, i-1),
                'k': random.choice([1, 3, 5]),
                's': random.choice([1, 2]),
                'c': random.choice([16, 32, 64])
            }
            props.append(p)
        return props

    def check_validity(self, arch_props: List[Dict[str, Any]]) -> bool:
        """
        Check if the random graph satisfies Shape constraints.
        Mirroring the logic encoded in SMT.
        """
        # Node 0 Input
        shapes = {0: (3, self.input_res, self.input_res)}
        
        for props in arch_props:
            node_id = props['id']
            op = props['op']
            in1 = props['in1']
            
            if in1 not in shapes: return False
            in_c, in_h, in_w = shapes[in1]
            
            if op == 1: # CONV
                k, s = props['k'], props['s']
                # Valid config?
                if (in_h + 2 - k) < 0: return False
                out_h = (in_h + 2 - k) // s + 1
                out_w = (in_w + 2 - k) // s + 1
                shapes[node_id] = (props['c'], out_h, out_w)
                
            elif op == 2: # POOL
                if in_h < 2: return False
                shapes[node_id] = (in_c, in_h//2, in_w//2)
                
            elif op == 3: # ADD
                in2 = props['in2']
                if in2 not in shapes: return False
                in2_c, in2_h, in2_w = shapes[in2]
                
                # Strict Match
                if in_c != in2_c or in_h != in2_h or in_w != in2_w:
                    return False
                shapes[node_id] = shapes[in1]
                
            elif op == 4: # CONCAT
                in2 = props['in2']
                if in2 not in shapes: return False
                in2_c, in2_h, in2_w = shapes[in2]
                
                # Height/Width Match
                if in_h != in2_h or in_w != in2_w:
                    return False
                shapes[node_id] = (in_c + in2_c, in_h, in_w)
                
        # If we reached here, it's valid shape-wise.
        # Check connectivity? (No Dead Ends). 
        # Simplified: Just checking if it runs is enough for baseline.
        return True

    def search(self, num_trials: int = 1000) -> Dict[str, Any]:
        """
        Run search. 
        Returns stats and the first valid architecture found (if any).
        """
        valid_count = 0
        first_valid = None
        start = time.time()
        
        for _ in range(num_trials):
            props = self.generate_random_props()
            if self.check_validity(props):
                valid_count += 1
                if not first_valid:
                    first_valid = props
                    
        duration = time.time() - start
        return {
            "trials": num_trials,
            "valid_count": valid_count,
            "valid_rate": valid_count / num_trials,
            "duration": duration,
            "first_valid_arch": first_valid
        }

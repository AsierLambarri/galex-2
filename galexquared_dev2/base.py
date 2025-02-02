import numpy as np

from .class_methods import gram_schmidt




class BaseHaloObject:
    """BaseSimulationObject that contains information shared between all objects in a simulation.
    """
    def __init__(self):
        self._parent = None  
        
        self.los = [1, 0, 0]
        self.basis = np.identity(3)
        self._old_to_new_base = np.identity(3)
        self.bound = False
        self.bound_method = None



    def set_parent(self, parent):
        """Sets the parent of this object and ensures attributes propagate from parent to child.
        """
        self._parent = parent
        if self._parent:
            if self.units is None:
                self.units = self._parent.units
            if self.basis is None:
                self.basis = self._parent.basis
    
        return None 
            
    def _set_los(self, los):
        """Sets the coordinate basis for this object and propagates to children if any.
        """
        self.los = los
        self._old_to_new_base = np.linalg.inv(gram_schmidt(los)) @ self.basis
        self.basis = gram_schmidt(los)

        if self._parent:
            self._parent._set_los(los)   

        return None
    
    def _switch_to_bound(self):
        self.bound = True
        
    def _switch_to_all(self):
        self.bound = False
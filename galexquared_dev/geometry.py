import numpy as np

from .class_methods import gram_schmidt




class Geometry:
    """BaseSimulationObject that contains information shared between all objects in a simulation.
    """
    def __init__(self):
        self._parent = None  
        
        self.los = [1, 0, 0]
        self.basis = np.identity(3)
        self._old_to_new_base = np.identity(3)



    def set_parent(self, parent):
        """Sets the parent of this object and ensures attributes propagate from parent to child.
        """
        self._parent = parent
        if self._parent:
            if self.basis is None:
                self.basis = self._parent.basis
    
            
    def _set_los(self, los):
        """Sets the coordinate basis for this object and propagates to children if any.
        """
        self.los = los
        self._old_to_new_base = np.linalg.inv(gram_schmidt(los)) @ self.basis
        self.basis = gram_schmidt(los)

        if self._parent:
            self._parent._set_los(los)   


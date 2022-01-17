import logging

import dolfinx.io


class XDMFFile(dolfinx.io.XDMFFile):

    KEYS_OF_MESHTAGS = 'KeysOfMeshTags'

    def write_mesh_meshtags(self, mesh, mts=None):
        """Write mesh and meshtags to XDMFFile

        Parameters
        ----------
        mesh:
            The dolfin mesh
        mts: optional
            The dict of MeshTags

        """
        logger = logging.getLogger("dolfiny")

        logger.info("Writing mesh")
        self.write_mesh(mesh)

        if mts is None:
            return

        keys_meshtags = {key: mt.dim for key, mt in mts.items()}

        logger.info("Writing information")
        self.write_information(self.KEYS_OF_MESHTAGS, str(keys_meshtags))

        logger.info("Writing meshtags")
        for mt in mts.values():
            mesh.topology.create_connectivity(mt.dim, mesh.topology.dim)
            self.write_meshtags(mt)
    

    def create_connectivity_all(self,mesh):
        # create all entities
        for d in range(0, mesh.topology.dim+1):
            mesh.topology.create_entities(d)
        for  d0 in range(0, mesh.topology.dim+1):
            for  d1 in range(0, mesh.topology.dim+1):
                mesh.topology.create_connectivity(d0, d1)
        return mesh 

    def read_mesh_meshtags(self, mesh_name='mesh'):
        """Read mesh and meshtags from XDMFFile

        Parameters
        ----------
        mesh_name: optional
            Name of the Grid node containing the mesh data in XDMF file

        Returns
        -------
        mesh:
            The dolfin mesh
        mts:
            The dict of meshtags

        """

        logger = logging.getLogger("dolfiny")

        logger.info("Reading mesh")
        mesh = self.read_mesh(name=mesh_name)
        # not supported any more

        #mesh.topology.create_connectivity_all()
        mesh = self.create_connectivity_all(mesh)

        logger.info("Reading information")
        value = self.read_information(self.KEYS_OF_MESHTAGS)

        import ast
        keys_meshtags = ast.literal_eval(value)

        mts = {}

        logger.info("Reading meshtags")
        for key in keys_meshtags.keys():
            mts[key] = self.read_meshtags(mesh, key)

        return mesh, mts

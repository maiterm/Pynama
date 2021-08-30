from petsc4py import PETSc
from matrices.mat_fs import MatFS
import numpy as np

class MatFSNS(MatFS):
    bcType ="FS-NS"
    def preAlloc_Kfs_Krhsfs(self, ind_d, ind_o , globalNodesDIR, globalNodesNS):
        dim = self.dom.getDimension()
        nodeStart, nodeEnd = self.dom.getNodesRange()

        locElRow = nodeEnd - nodeStart
        vel_dofs = locElRow * dim

        dns_nnz = np.zeros(locElRow)
        ons_nnz = np.zeros(locElRow)
        drhsns_nnz = np.zeros(locElRow)
        orhsns_nnz = np.zeros(locElRow)


        for node, connect in enumerate(ind_d):
            if (node + nodeStart) not in (globalNodesNS| globalNodesDIR ):
                dns_nnz[node] = len(connect & globalNodesNS)
            elif (node + nodeStart) in globalNodesNS:
                # FIXME: len() can be distributed on each set operation
                dns_nnz[node] = len((connect - globalNodesDIR) |
                                  (connect & globalNodesNS))
            #drhsns_nnz[node] = len(connect & globalNodesNS)

        for node, connect in enumerate(ind_o):
            if (node + nodeStart) not in (globalNodesNS| globalNodesDIR ):
                ons_nnz[node] = len(connect & globalNodesNS)
            elif (node + nodeStart) in globalNodesNS:
                ons_nnz[node] = len((connect - globalNodesDIR) |
                                  (connect & globalNodesNS))

            #orhsns_nnz[node] = len(connect & globalNodesNS)



        for node, connect in enumerate(ind_d):
            if (node + nodeStart) in (globalNodesDIR - globalNodesNS):
                drhsns_nnz[node] = 1
            else:
                drhsns_nnz[node] = len(connect & (globalNodesDIR | globalNodesNS))

        for node, connect in enumerate(ind_o):
                orhsns_nnz[node] = len(connect & (globalNodesDIR | globalNodesNS))

        drhsns_nnz_ind, orhsns_nnz_ind = self.createNNZWithArray(drhsns_nnz,orhsns_nnz, dim, dim)
        dns_nnz_ind, ons_nnz_ind =self.createNNZWithArray(dns_nnz, ons_nnz, dim, dim)

        self.Kfs = self.createEmptyMat(vel_dofs,vel_dofs,dns_nnz_ind, ons_nnz_ind)
        self.Kfs.setName("Kfs")
        self.kle.append(self.Kfs)

        self.Krhsfs = self.createEmptyMat(vel_dofs,vel_dofs, drhsns_nnz_ind, orhsns_nnz_ind)
        self.Krhsfs.setName("Krhsfs")
        self.kle.append(self.Krhsfs)

    def buildFSNS(self,globNodesDirichlet):
        indices2one = set() 
        indices2onefs = set()
        cellStart , cellEnd = self.dom.getLocalCellRange()
        globalTangIndicesNS = self.dom.getTangDofs(collect=True)
        globalNormalIndicesNS = self.dom.getNormalDofs(collect=True)
        dim = self.dom.getDimension()
        

        
        for cell in range(cellStart, cellEnd):
            nodes , inds , localMats = self.dom.computeLocalKLEMats(cell)
            locK, locRw, locRd = localMats
            indicesVel, indicesW = inds

            indicesVelSet = set(indicesVel)
            normalDofs = globalNormalIndicesNS & indicesVelSet
            tangentialDofs = globalTangIndicesNS & indicesVelSet
            normalDofs -= tangentialDofs 

            nodeBCintersect = set(globNodesDirichlet) & set(nodes)
            glDirDofs =  [n*dim + dof for n in nodeBCintersect for dof in range(dim)]
            locDirDofs = list()
            for node in nodeBCintersect:
                localBoundaryNode = nodes.index(node)
                # FIXME : No importa el bc, #TODO cuando agregemos NS si importa
                for dof in range(dim):
                    locDirDofs.append(localBoundaryNode*dim + dof)
            
            gldofSetFSNS = list(normalDofs | set(glDirDofs) )
            gldofFreeFSSetNS = list(tangentialDofs )
            gldof2beSet = list(normalDofs | tangentialDofs | set(glDirDofs)) #[indicesVel[ii] for ii in dof2beSet]
            gldofFree = list(indicesVelSet - set(gldof2beSet))
            
            dofFree = [ indicesVel.index(i) for i in gldofFree ]
            locNormalDof = [ indicesVel.index(i) for i in normalDofs ]
            locTangDof = [ indicesVel.index(i) for i in tangentialDofs ]

            dofFreeFSSetNS = locTangDof
            dofSetFSNS = list(set(locNormalDof) | set(locDirDofs) ) 
            dof2beSet = list(set(dofFreeFSSetNS) | set(dofSetFSNS) )


            if normalDofs | tangentialDofs | set(glDirDofs) :
                self.Krhs.setValues(
                gldofFree, gldof2beSet,
                -locK[np.ix_(dofFree, dof2beSet)], addv=True)
                indices2one.update(gldof2beSet)

                # FIXME: is the code below really necessary?
                for indd in gldof2beSet:
                    self.Krhs.setValues(indd, indd, 0, addv=True)
                self.Kfs.setValues(gldofFreeFSSetNS, gldofFree,
                                    locK[np.ix_(dofFreeFSSetNS, dofFree)],
                                    addv=True)

                self.Kfs.setValues(gldofFree, gldofFreeFSSetNS,
                                    locK[np.ix_(dofFree, dofFreeFSSetNS)],
                                    addv=True)

                self.Kfs.setValues(
                    gldofFreeFSSetNS, gldofFreeFSSetNS,
                    locK[np.ix_(dofFreeFSSetNS, dofFreeFSSetNS)],
                    addv=True)

                # Indices where diagonal entries should be reduced by 1
                indices2onefs.update(gldofFreeFSSetNS)

                self.Rwfs.setValues(gldofFreeFSSetNS, indicesW,
                                    locRw[dofFreeFSSetNS, :], addv=True)

                self.Rdfs.setValues(gldofFreeFSSetNS, nodes,
                                    locRd[dofFreeFSSetNS, :], addv=True)
                self.Krhsfs.setValues(
                        gldofFreeFSSetNS, gldofSetFSNS,
                        - locK[np.ix_(dofFreeFSSetNS, dofSetFSNS)], addv=True)
                self.Krhsfs.setValues(
                        gldofFree, gldofSetFSNS,
                        - locK[np.ix_(dofFree, dofSetFSNS)], addv=True)
                for indd in gldofSetFSNS:
                        self.Krhsfs.setValues(indd, indd, 0, addv=True)
            
        

            self.K.setValues(gldofFree, gldofFree,
                             locK[np.ix_(dofFree, dofFree)], addv=True)

            for indd in gldof2beSet:
                self.K.setValues(indd, indd, 0, addv=True)

            self.Rw.setValues(gldofFree, indicesW,
                              locRw[np.ix_(dofFree, range(len(indicesW)))], addv=True)

            self.Rd.setValues(gldofFree, nodes,
                              locRd[np.ix_(dofFree, range(len(nodes)))],
                              addv=True)
        
        self.assembleAll()
        self.setIndices2One(indices2one)

        for indd in indices2onefs:
            self.Kfs.setValues(indd, indd, -1, addv=True)

        self.Kfs.assemble()
        self.Rwfs.assemble()
        self.Rdfs.assemble()
        self.Krhsfs.assemble()
        
        for indd in (indices2one - indices2onefs):
            self.Krhsfs.setValues(indd, indd, 1, addv=False)
        self.Krhsfs.assemble()

        if not self.comm.rank:
            self.logger.info(f"KLE Matrices builded")
                                                                  

    def preAlloc_K_Krhs(self, ind_d, ind_o, d_nnz, o_nnz, locIndicesDir ,globalNodesDir,globalNodesNS):
        dim = self.dom.getDimension()
        nodeStart, nodeEnd = self.dom.getNodesRange()

        locElRow = nodeEnd - nodeStart
        vel_dofs = locElRow * dim

        drhs_nnz = np.zeros(locElRow)
        orhs_nnz = np.zeros(locElRow)

        for node, connectivity in enumerate(ind_d):
            if (node + nodeStart) in globalNodesDir:
                drhs_nnz[node] = 1
            else:
                drhs_nnz[node] = len(connectivity & (globalNodesDir | globalNodesNS))
                d_nnz[node] = d_nnz[node] - len(connectivity & globalNodesDir )
                
        for node, connectivity in enumerate(ind_o):
            orhs_nnz[node] = len(connectivity &( globalNodesDir | globalNodesNS))

        d_nnz_ind, o_nnz_ind = self.createNNZWithArray(d_nnz, o_nnz, dim, dim )
        drhs_nnz_ind, orhs_nnz_ind = self.createNNZWithArray(drhs_nnz, orhs_nnz, dim, dim )

        d_nnz_ind[locIndicesDir] = 1
        o_nnz_ind[locIndicesDir] = 0

        self.K = self.createEmptyMat(vel_dofs, vel_dofs, d_nnz_ind, o_nnz_ind)
        self.K.setName("K")
        self.Krhs = self.createEmptyMat(vel_dofs, vel_dofs, drhs_nnz_ind, orhs_nnz_ind)
        self.Krhs.setName("Krhs")
        self.kle.append(self.K)
        self.kle.append(self.Krhs)

    def preAlloc_Rd_Rw(self, diag_nnz, off_nnz, locIndicesDir,locIndicesNS, createFS=False):
        dim, dim_w, _ = self.dom.getDimensions()
        locElRow = len(diag_nnz)
        vel_dofs = locElRow * dim
        vort_dofs = locElRow * dim_w

        dw_nnz_ind, ow_nnz_ind = self.createNNZWithArray(diag_nnz, off_nnz, dim_w, dim)
        dd_nnz_ind, od_nnz_ind = self.createNNZWithArray(diag_nnz, off_nnz, 1, dim)

        
        dwns_nnz_ind = np.zeros(locElRow * dim, dtype=np.int32)
        owns_nnz_ind = np.zeros(locElRow * dim, dtype=np.int32)

        ddns_nnz_ind = np.zeros(locElRow * dim, dtype=np.int32)
        odns_nnz_ind = np.zeros(locElRow * dim, dtype=np.int32)

        dwns_nnz_ind[locIndicesNS] = dw_nnz_ind[locIndicesNS]
        owns_nnz_ind[locIndicesNS] = ow_nnz_ind[locIndicesNS]
                                
        ddns_nnz_ind[locIndicesNS] = dd_nnz_ind[locIndicesNS]
        odns_nnz_ind[locIndicesNS] = od_nnz_ind[locIndicesNS]

        self.Rwfs =self.createEmptyMat(vel_dofs,vort_dofs, dwns_nnz_ind, owns_nnz_ind)
        self.Rwfs.setName("Rwfs")
        self.kle.append(self.Rwfs)
        self.Rdfs =self.createEmptyMat(vel_dofs,locElRow, ddns_nnz_ind, odns_nnz_ind)
        self.Rdfs.setName("Rdfs")
        self.kle.append(self.Rdfs)

        dw_nnz_ind[locIndicesDir] = 0
        dw_nnz_ind[locIndicesNS] = 0
        ow_nnz_ind[locIndicesDir] = 0
        ow_nnz_ind[locIndicesNS] = 0
                  
        dd_nnz_ind[locIndicesNS] = 0
        od_nnz_ind[locIndicesNS] = 0

        self.Rw = self.createEmptyMat(vel_dofs, vort_dofs, dw_nnz_ind, ow_nnz_ind)
        self.Rw.setName("Rw")
        self.Rd = self.createEmptyMat(vel_dofs, locElRow, dd_nnz_ind, od_nnz_ind)
        self.Rd.setName("Rd")
        self.kle.append(self.Rw)
        self.kle.append(self.Rd)

    def build(self, buildKLE=True, buildOperators=True):
        locNodesNS = np.array(list(self.dom.getNodesNoSlip()))
        nodeStart, _ = self.dom.getNodesRange()
        locNodesNS -= nodeStart
        globNodesNS = self.dom.getNodesNoSlip(collect=True)

        locNodesDirichlet = np.array(list(self.dom.getNodesDirichlet()))
        nodeStart, _ = self.dom.getNodesRange()
        locNodesDirichlet -= nodeStart
        globNodesDirichlet = self.dom.getNodesDirichlet(collect=True)

        dim = self.dom.getDimension()
        locIndNS = [ node*dim+dof for node in locNodesNS for dof in range(dim) ]
        conn_diag, conn_offset, nnz_diag, nnz_off = self.dom.getConnectivity()


        locIndDirichlet = [ node*dim+dof for node in locNodesDirichlet for dof in range(dim) ]
        


        if buildOperators:
            self.preAlloc_operators(nnz_diag, nnz_off)
        if buildKLE:
            self.preAlloc_Rd_Rw(nnz_diag, nnz_off,locIndDirichlet, locIndNS, createFS=True)
            self.preAlloc_K_Krhs(conn_diag, conn_offset, nnz_diag, nnz_off, locIndDirichlet, globNodesDirichlet, globNodesNS)
            self.preAlloc_Kfs_Krhsfs(conn_diag, conn_offset, globNodesDirichlet, globNodesNS)

        self.buildFSNS(globNodesDirichlet)
        self.buildOperators()

from petsc4py.PETSc import IS, Vec, COMM_WORLD
import numpy as np
import logging
from math import radians, sin, cos

from .boundary import Boundary, FunctionBoundary

class BoundaryConditions:
    types = ["FS", "NS", "FS-NS"]
    bcTypesAvailable = ("uniform", "custom-func", "free-slip", "no-slip")
    comm = COMM_WORLD
    def __init__(self, sides: list):
        self.__boundaries = list()
        self.__nsBoundaries = list()
        self.__fsBoundaries = list()
        self.__type = None
        self.__ByName = dict()
        self.__ByType = { "free-slip": [], "no-slip": []}
        self.__needsCoords = list()
        self.__borderNames = sides
        self.__dim = 2 if len(sides) == 4 else 3

        self.logger = logging.getLogger(f"[{self.comm.rank}]Boundary Conditions:")

    def __repr__(self):
        txt = " --== Boundary Conditions ==--\n"
        txt += "   Name   |   Type   |   Values   |   Nodes   \n"
        for b in self.__boundaries:
            name = b.getName()
            typ = b.getType()
            try:
                val = b.getVelocitySetted()
            except:
                val = "Not defined"
            try:
                dirs = b.getNodes()
                dirs = str(dirs)
            except:
                dirs = "Not Defined"
            msg = f"{name:10}|{typ:10}|{str(val):12}|{dirs:12}\n"
            txt+=msg
        return txt

    def setBoundaryConditions(self, data):
        # data its the dictionary with key 'boundary-conditions'
        if "uniform" in data:
            if "free-slip" in data or "no-slip" in data:
                self.logger.warning("WARNING: Only constant bc its assumed")
            self.__type = "FS"
            vals = self.__handleUniform(data['uniform'])
            self.__setUpBoundaries('free-slip', self.__borderNames, vals)
        elif "custom-func" in data and "no-slip" in data:
            self.__type = "FS-NS"
            self.__setFunctionBoundaries( data['custom-func']['name'], data['custom-func']['attributes'], data['custom-func']['borders-name'])
            self.__setPerBoundaries('no-slip', data['no-slip'])
        elif "custom-func" in data:
            self.__type = "FS"
            funcName = data['custom-func']['name']
            attrs = data['custom-func']['attributes']
            self.__setFunctionBoundaries(funcName, attrs)
        elif "free-slip" in data:
            self.__type = "FS"
            self.__setPerBoundaries('free-slip', data['free-slip'])
        elif "no-slip" in data:
            self.__type = "NS"
            self.__setPerBoundaries('no-slip', data['no-slip'])
        else:
            raise Exception("Boundary Conditions not defined")
        print(self.__type)

    def getType(self):
        return self.__type

    def __setUpBoundaries(self, t, sides, vals: dict):
        for nameSide in sides:
            self.__setBoundary(nameSide, t, vals)

    def __setPerBoundaries(self, t, sidesDict: dict):
        for name, vals in sidesDict.items():
            self.__setBoundary(name, t , vals)


    def __handleUniform(self, dataUniform: dict):
        out = dict()
        if "velocity" in dataUniform and "vorticity" not in dataUniform:
            out['velocity']  = dataUniform['velocity']
            out['vorticity'] = [0]
        elif "re" in dataUniform:
            try:
                assert "mu" in dataUniform
                assert "rho" in dataUniform
                assert "Lref" in dataUniform
                assert "direction" in dataUniform
            except:
                raise Exception("mu, rho, Lref AND/OR direction not defined")

            re = dataUniform['re']
            L = eval(dataUniform['Lref'])
            mu = dataUniform['mu']
            rho = dataUniform['rho']
            direction = dataUniform['direction']
            angleRadian = radians(direction)
            velRef = re*(mu/rho) / L
            out['velocity'] = [cos(angleRadian)*velRef,sin(angleRadian)*velRef]
            out['vorticity'] = [0]

        else:
            out = dataUniform

        return out

    def __setBoundary(self, name, typ, vals: dict):
        boundary = Boundary(name, typ, self.__dim)

        if type(vals) == list:
            boundary.setValues('velocity', vals)
            boundary.setValues('vorticity', [0] if self.__dim==2 else [0]*3)
            print(typ,name,vals)
        else:
            for attrName, val in vals.items():
                boundary.setValues(attrName, val)

        self.__boundaries.append(boundary)
        if typ == 'free-slip':
            self.__fsBoundaries.append(boundary)
        elif typ == 'no-slip':
            self.__nsBoundaries.append(boundary)
        else:
            raise Exception("Wrong boundary type")
        self.__ByType[typ].append(boundary)
        self.__ByName[name] = boundary

    def __setFunctionBoundaries(self, funcName, attrs,borderNames = None):
        if not borderNames:
            borderNames = self.__borderNames
        for borderName in borderNames:
            self.__setFunctionBoundary(borderName, funcName, attrs)

    def __setFunctionBoundary(self, borderName, funcName, attrs):
        dim = self.__dim
        boundary = FunctionBoundary(borderName , funcName , attrs , dim)
        self.__boundaries.append(boundary)
        self.__fsBoundaries.append(boundary)
        self.__ByName[borderName] = boundary
        self.__ByType['free-slip'].append(boundary)
        self.__needsCoords.append(borderName)

    def getNames(self, bcs=None):
        if bcs == None:
            bcs = self.__boundaries
        bNames = list()
        for b in bcs:
            bNames.append(b.getName()) 
        return bNames

    def getNamesByType(self, bcType):
        bcs = self.__ByType[bcType]
        return self.getNames(bcs)

    def getBordersNeedsCoords(self):
        return self.__needsCoords

    def getIndicesByName(self, name):
        border = self.__ByName[name]
        return border.getDofsConstrained()

    def setBoundaryNodes(self, bName, nodes):
        try:
            boundary = self.__ByName[bName]
            boundary.setNodes(nodes)
        except:
            raise Exception("Boundary Not found")

    def setBoundaryCoords(self, bName, coords):
        border = self.__ByName[bName]
        border.setNodesCoordinates(coords)

    def getIndicesByType(self, bcType):
        inds = IS().createGeneral([])
        boundaries = self.__ByType[bcType]
        if len(boundaries) == 0:
            return set()
        else:
            for bc in self.__ByType[bcType]:
                bcIS = bc.getIS()
                inds = bcIS.union(inds)
            return set(inds.getIndices())

    def getNodesByType(self, bcType, allGather=False):
        inds = set()
        boundaries = self.__ByType[bcType]
        if len(boundaries) == 0:
            return set()
        else:
            for bc in self.__ByType[bcType]:
                bcIS = bc.getIS()
                locIndices = set(bcIS.getBlockIndices())
                if allGather:
                    collectIndices = self.comm.tompi4py().allgather([locIndices])
                    for remoteIndices in collectIndices:
                        locIndices |= remoteIndices[0]
                else:
                    pass
                inds |= locIndices
            return inds

    def getNoSlipIndices(self, allGather=False):
        inds = IS().createGeneral([])
        for bc in self.__nsBoundaries:
            bcIS = bc.getIS()
            inds = bcIS.union(inds)
        return set(inds.getIndices())

    def getNoSlipTangDofs(self, allGather=False):
        dofs = set()
        for bc in self.__nsBoundaries:
            locTang = bc.getTangDofs()
            if allGather:
                collectIndices = self.comm.tompi4py().allgather(locTang)
                for remoteIndices in collectIndices:
                    locTang |= remoteIndices
            dofs |= locTang
        return dofs

    def getNoSlipNormalDofs(self, allGather=False):
        dofs = set()
        removeSet = set()
        for bc in self.__nsBoundaries:
            locDofs = bc.getNormalDofs()
            if allGather:
                collectIndices = self.comm.tompi4py().allgather(locDofs)
                for remoteIndices in collectIndices:
                    if bc.getName() in ["left","right"]:
                        remoteIndicesT = remoteIndices.copy()
                        for ind in remoteIndicesT:
                            if (ind+1) in dofs:
                                removeSet.add(ind)
                    locDofs |= remoteIndices
            dofs |= locDofs
        dofs-=removeSet
        return dofs

    def getFreeSlipIndices(self):
        inds = IS().createGeneral([])
        for bc in self.__fsBoundaries:
            bcIS = bc.getIS()
            inds = bcIS.union(inds)
        return set(inds.getIndices())

    def setValuesToVec(self, vec, name, t, nu):
        boundaries = self.__boundaries 

        if name == 'velocity':
            boundaries = self.__fsBoundaries
            for b in self.__nsBoundaries:
                normalDir = b.getNormalDirection()
                vel = 0
                numOfNodes = b.getNumOfNodes()
                indsNormal = b.getNormalDofs()
                collectIndices = self.comm.tompi4py().allgather(indsNormal)   
                velNormal = np.repeat(vel, numOfNodes)
                vec.setValues(list(indsNormal), velNormal , addv=False)
            
        for b in boundaries:
            arr = b.getValues(name, t, nu)
            if self.__dim == 2 and name == 'vorticity':
                inds = b.getNodes()
            else:
                inds = b.getDofsConstrained()
            vec.setValues(inds, arr, addv=False)
            vec.assemble()

            # for b in self.__nsBoundaries:
            #     numOfNodes = b.getNumOfNodes()
            #     vel = 0
            #     indsNormal = b.getDofsConstrained()
            #     velNormal = np.repeat(vel, numOfNodes)
            #     vec.setValues(list(indsNormal), velNormal , addv=False)

        vec.assemble()

    def setTangentialValuesToVec(self, vec, name, t, nu):
        """This method is useful to impose the no slip condition"""
        for b in self.__boundaries:
            arr = b.getValues(name, t, nu)
            inds = b.getDofsConstrained()
            vec.setValues(inds, arr, addv=False)
            vec.assemble()
        for bc in self.__nsBoundaries:
            tangDirs = bc.getTangDirections()
            vel = bc.getVelocitySetted()
            numOfNodes = bc.getNumOfNodes()  
               
            for tang in tangDirs:
                indsTang = bc.getTangDofs(tang)
                collectIndices = self.comm.tompi4py().allgather(indsTang)
                
                velTang = np.repeat(vel[tang], numOfNodes)
                vec.setValues(list(indsTang), velTang , addv=False)

        vec.assemble()
        #bcIS = bc.getIS()

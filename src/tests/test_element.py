import unittest
import numpy as np
import math
import numpy.testing as np_test
from domain.elements.utilities import gaussPoints, lobattoPoints, GaussPoint2D, generateGaussPoints3D ,GaussPoint3D ,generateGaussPoints2D
from domain.elements.spectral import Spectral


class UtilitiesElementTest(unittest.TestCase):
    def test_generate_list_gausspoints2D(self):
        for i_test in range(2,5):
            with self.subTest(i_test=i_test):
                # my new gausspoint list generator func
                gps_1D, gps_wei = gaussPoints(i_test)
                gps_list = generateGaussPoints2D(gps_1D, gps_wei)
                legacy_list = list()
                
                # this implementation was made by Alejandro
                for i in range(len(gps_1D)):
                    for j in range(len(gps_1D)):
                        legacy_list.append(GaussPoint2D(gps_1D[i], gps_1D[j], gps_wei[i]*gps_wei[j]))
                
                for i in range(len(gps_1D)**2):
                    with self.subTest(i=i):
                        gps_test = gps_list[i]
                        gps_exact = legacy_list[i]
                        np_test.assert_array_almost_equal(gps_exact.r , gps_test.r)
                        np_test.assert_array_almost_equal(gps_exact.s , gps_test.s)
                        np_test.assert_array_almost_equal(gps_exact.w , gps_test.w)


    def test_generate_list_gausspoints3D(self):
        for i_test in range(2,5):
            with self.subTest(i_test=i_test):
                # my new gausspoint list generator func
                gps_1D, gps_wei = gaussPoints(i_test)
                gps_list = generateGaussPoints3D(gps_1D, gps_wei)
                # this implementation was made by Alejandro
                legacy_list = list()
                for i in range(len(gps_1D)):
                    for j in range(len(gps_1D)):
                        for k in range(len(gps_1D)):
                            legacy_list.append(GaussPoint3D(gps_1D[i], gps_1D[j],
                            gps_1D[k], gps_wei[i]*gps_wei[j]*gps_wei[k]))
                
                for i in range(len(gps_1D)**3):
                    with self.subTest(i=i):
                        gps_test = gps_list[i]
                        gps_exact = legacy_list[i]
                        np_test.assert_array_almost_equal(gps_exact.r , gps_test.r)
                        np_test.assert_array_almost_equal(gps_exact.s , gps_test.s)
                        np_test.assert_array_almost_equal(gps_exact.t , gps_test.t)
                        np_test.assert_array_almost_equal(gps_exact.w , gps_test.w)

    def test_gausspoints_N2(self):
        # Seteo de valores exactos
        weights = np.array([1, 1], dtype=float)

        gp1 = 1 / math.sqrt(3)
        gps = np.array([ -gp1, gp1  ], dtype=float)

        # Generacion a partir de la funcion a testear
        gps_result, weights_result = gaussPoints(2)
        np_test.assert_array_equal(gps, gps_result)
        np_test.assert_array_equal(weights, weights_result) 

    def test_gausspoints_N3(self):
        # Seteo de valores exactos
        weights = np.array([ 5/9 , 8/9 , 5/9], dtype=float)

        gp1 = math.sqrt(3/5)
        gps = np.array([ -gp1, 0 ,gp1  ], dtype=float)

        # Generacion a partir de la funcion a testear
        gps_result, weights_result = gaussPoints(3)
        np_test.assert_array_almost_equal(gps, gps_result, decimal=12)
        np_test.assert_array_almost_equal(weights, weights_result, decimal=12) 

    def test_lobattopoints(self):
        # Get the exact lobatto points
        lob_poi_2 = np.array([-1, 1], dtype=float)
        wei_2 = np.array([1, 1], dtype=float)

        lob_poi_3 = np.array([-1, 0, 1], dtype=float)
        wei_3 = np.array([ 1/3, 4/3, 1/3], dtype=float)

        lob_poi_4 = np.array([-1, -1/math.sqrt(5), 1/math.sqrt(5),1], dtype=float)
        wei_4 = np.array([1/6,5/6,5/6,1/6])

        lobpoi = [lob_poi_2, lob_poi_3, lob_poi_4]
        weights = [wei_2, wei_3, wei_4]

        for i in range(3):
            with self.subTest(i=i):
                lobpoi_calculated, wei_calulated = lobattoPoints(i+2)
                np_test.assert_array_almost_equal(
                weights[i]
                ,wei_calulated
                ,decimal=12)
                np_test.assert_array_almost_equal(
                lobpoi[i]
                ,lobpoi_calculated 
                ,decimal=12)

class SpectralTestNodes(unittest.TestCase):
    def setUp(self):
        self.spectral_elements_2d = list()
        self.spectral_elements_3d = list()
        for ngl in range(2,5):
            self.spectral_elements_2d.append(Spectral(ngl,2))
            self.spectral_elements_3d.append(Spectral(ngl,3))

    def test_nnodes(self):
        for i, spectral in enumerate(self.spectral_elements_2d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( (i+2) ** 2 , spectral.nnode)
        for i, spectral in enumerate(self.spectral_elements_3d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual((i+2) ** 3 , spectral.nnode)

    def test_edges_nodes(self):
        for i, spectral in enumerate(self.spectral_elements_2d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( i , spectral.nnodedge)
        for i, spectral in enumerate(self.spectral_elements_3d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( i , spectral.nnodedge)

    def test_faces_nodes(self):
        for i, spectral in enumerate(self.spectral_elements_3d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( i**2 , spectral.nnodface)

    def test_body_nodes(self):
        for i, spectral in enumerate(self.spectral_elements_2d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( i**2 , spectral.nnodcell)
        for i, spectral in enumerate(self.spectral_elements_3d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( i**3 , spectral.nnodcell)

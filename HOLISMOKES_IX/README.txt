**************************************************************************************************
**************************************************************************************************
**												**
**   README file for the code apply_net.py which applies the trained network to the		**
**   provided HSC image cutouts.			      	      	  	  	     	**
**												**
**   Last edit: Feb. 03, 2023		 	       		     	    	  		**
**   Written by Dr. Stefan Schuldt								**
**   Comments and questions: stefan.schuldt@unimi.it						**
**												**
**************************************************************************************************
**************************************************************************************************

We provide here the trained network from our paper "HOLISMOKES -- IX. Neural network inference of strong-lens parameters and uncertainties from ground-based images" (Schuldt et al. 2022) and a code to apply it. It is trained for galaxy-scale lens images observed with the Hyper-Suprime Cam (HSC) in griz bands, and thus this network should be only applied to such kind of lenses. The required python modules, files, and code usage are described below. Please check out the corresponding publication for further details and reach out to us in case of any questions.


**************** CODE REFERENCE *****************

If you make use of this code, please cite:
Schuldt, S., Canameras, R., Shu, Y., et al. 2022, arXiv e-prints, arXiv:2206.11279
https://ui.adsabs.harvard.edu/abs/2022arXiv220611279S/abstract

@ARTICLE{2022arXiv220611279S,
       author = {{Schuldt}, S. and {Ca{\~n}ameras}, R. and {Shu}, Y. and {Suyu}, S.~H. and {Taubenberger}, S. and {Meinhardt}, T. and {Leal-Taix{\'e}}, L.},
        title = "{HOLISMOKES -- IX. Neural network inference of strong-lens parameters and uncertainties from ground-based images}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2022,
        month = jun,
          eid = {arXiv:2206.11279},
        pages = {arXiv:2206.11279},
          doi = {10.48550/arXiv.2206.11279},
archivePrefix = {arXiv},
       eprint = {2206.11279},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220611279S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


**************** PROVIDED FILES *********************

We provide the following files:

- README.txt	 : This readme file containing an overview of the code and how to use it.
- IDs.txt	 : Example list of the lens IDs.
- apply_net.py	 : code that loads the trained network and apply it to the provided lens images.
- final.t7	 : saved final network presented in Schuldt et al. (2022)
- images	 : folder containing 16 example mock images displayed in Fig. 2 of Schuldt et al. (2022)
- pred_median.npy: numpy array containing the median network predictions of the 16 example images.
- pred_sigma.npy : numpy array containing the sigma network predictions of the 16 example images.

**************** REQUIRED SOFTWARE **************

To run the code, please make sure you have installed python3 and the following python modules:

- numpy=1.22.0
- random=1.0.1
- astropy=5.0
- troch=1.11.0

The above module versions correspond to those used for code testing. We recommend to have at least the same version.

**************** CODE USAGE *************************


The code accepts the following arguments:

-cat	 : required, path to the text-file containing the IDs of your lenses
-data	 : required, path to images
-o	 : optional, name of output catalog
-col	 : optional, column number of table that contains the IDs
-print	 : optional, if used values of each lens get printed on the terminal.

example command to call the code:

	python3 apply_net.py -cat IDs.txt -data images

*************** DATA FORMAT *************************

The following files need to be updated if you would like to apply the network to other HSC lenses.

- IDs.txt : This file contains alphanumerical identifications of each lens. It need to be a plain text file without header. If there are more columns separated by a space, you can use the -col argument to specify the column.

- images/ : This is the folder containing your images of the lenses. The name of the folder is free and need to be provided through the -data argument.

- lens_ID.fits : These are the actual images of your lenses, all stored in a single folder (see above). Each lens correspond to a single file, i.e. a fits data cube including the g, r, i, and z bands from HSC. Note that the order of filters is crucial. The images must be 64x64 pixels, with a pixel size of 0.168 arcsec. The nomenclature is: "lens_ID.fits" while the alphanumerical ID is the corresponding identification number listed in IDs.txt.



*************** CODE OUTPUT *************************

By calling above command, the code will produce two files:

- pred_median.npy: numpy array containing the network predictions for the median parameter values of the 16 example images.
- pred_sigma.npy : numpy array containing the network predictions for the parameter uncertainties (1 sigma) of the 16 example images.

The values in those files correspond to the example images and are provided for a direct comparison and test of the code. Each file contain 7 values per lens:

     - x-coordinate of the lens center in arcsec (with respect to the image center)
     - y-coordinate of the lens center in arcsec (with respect to the image center)
     - x component of the ellipticity in complex notation
     - y component of the ellipticity in complex notation
     - Einstein radius in arcsec
     - x component of the external shear in complex notation
     - y component of the external shear in complex notation

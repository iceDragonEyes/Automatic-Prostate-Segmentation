/*=========================================================================
This code implements a totally automated algorithm for MRI prostate volume 
segmentation.
This code is completely developed by the biomedical engineer Stefano Ravetto,
committed by Polytechnic University of Turin and IRCCS (Institute for cancer 
research and threatment of Candiolo), using C++ and ITK (you need the ITK 
libraries and review to run this code).
Sadly, this code in no more under development, though many improvements can be
implemented.
For any information or if you are interested in further development of this code,
please contact me at: 
	 stefano.ravetto.bme@gmail.com
	 stefano.ravetto@studenti.polito.it
	 https://it.linkedin.com/pub/stefano-ravetto/38/289/671

To run this code you need:
· b0 DWI 2D slices of the volume of the prostate
· T2 2D slices of the volume of the prostate
· ROI boxes 2D slices of the volume of the prostate
(if you don't have the ROI the code can be easly fixed or the ROI easly
 obtained. See detailed description in the related part of the code)

The code expects the input images in the following way:
 · '~/reg_dwi/DWI1000/b0_reg_full', where inside must be stored the b0 DWI 2D 
	   images, in dicom format, and named as 'slice01.dcm, slice02.dcm, ....,
   slice10.dcm,...' 
 · '~/T2/dcm', where inside must be stored the T2 2D images, in dicom format
	   (no particular name is needed dicom information will be used instead
 · '~/reg_dwi/ROI_reg_full', where inside must be stored the ROI boxes 2D images,
	    in dicom format, and named as 'slice01.dcm, slice02.dcm, ....,
	    slice10.dcm,...' 

The code expects as input the patient main directory, where inside its first 
	    sub-directories must be the three described above.
For example, you need to have a directory tree like:
/patients/patientnumber1/reg_dwi/DWI1000/b0_reg_full
/patients/patientnumber1/T2/dcm
/patients/patientnumber1/reg_dwi/ROI_reg_full

'/patients/patientnumber1' is your core directory, in where all the other 
	    proccessing images and their directory will be created and saved.

Please note that this is the first implementation of this code and the first 
	    method to try to be fully automated.
The code is written trying to be as general, multipurpose, and with less fixed 
	    value as possible, but this is not guaranteed.
I encourage you to read the code to evaluate that any of its parts and any fixed
	    value fits your needs.

Usage: 
./LevelsetProstateSegmentation_Eval+Esc+H	   CoreDirectory		[output mode]
output mode is optional:
0 (default): verbose mode (it's a very verbose mode, usefull to understand the 
                           code behavior for debug mode, or for first time using)
1          : short mode, with few informations and progress bars

Here follows an index of the code parts:
01 - [l283] 		Initialization
02 - [l367]			dcm2nii_dwi - original b0 volume build
03 - [l472]			Original b0 volume write
04 - [l534] 		Gaussian Discrete 3D Filter
05 - [l594]			dcm2nii_dwi - ROI volume build
06 - [l691]			ROI volume write
07 - [l787]			ExtractImageWriter
08 - [l1134]		Non-prostatic Slice Exclusion Criteria
09 - [l2105]		Optimal Levelset Heaviside function value automatic search 
10 - [l2466]		Multiple Initialization parameters extraction
11 - [l3330]		Post-processing
12 - [l3704]		Superior Periferical Coil Tissue Segmentation
13 - [l4537]		Segmentation Refining
14 - [l4882]		dcm2nii_dwi - Canny volume
15 - [l4989]		3D Canny Levelset Segmantation
16 - [l5198]		Post_processing
17 - [l5465]		New Slices Creation
18 - [l5583]		dcm2nii_dwi - post-processed Canny 
19 - [l5697]		Resampling and Change Information to match T2 images
 =========================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

// dcm2niiImageFilter
#include "itkOrientedImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include <itkOrientImageFilter.h>

// ExtractImageFilter
#include "itkExtractImageFilter.h"
#include "itkImage.h"

// ChangeInformationFilter
#include "itkVersor.h"
#include "itkChangeInformationImageFilter.h"

// ChanAndVeseDenseLevelsetFilter
#include "itkScalarChanAndVeseDenseLevelSetImageFilter.h"
#include "itkScalarChanAndVeseLevelSetFunctionData.h"
#include "itkConstrainedRegionBasedLevelSetFunctionSharedData.h"
#include "itkFastMarchingImageFilter.h"
#include "itkAtanRegularizedHeavisideStepFunction.h"

// String Stream
#include <sstream>

// Time Probe
#include "itkTimeProbe.h"

// Create directory and files
#include "itkFileTools.h"

// MaskImageFilter
#include "itkMaskImageFilter.h"
#include "itkMaskNegatedImageFilter.h"


// Gaussian Filter
#include <itkDiscreteGaussianImageFilter.h>

// Crosscorrelation
#include "itkNormalizedCorrelationImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkImageKernelOperator.h"
#include "itkRescaleIntensityImageFilter.h"

// StatisticsImageFilter
#include "itkStatisticsImageFilter.h"

// Subsample Image Filter
#include "itkShrinkImageFilter.h"

// Some math operations
#include "numeric"

// Erosion Filter
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryBallStructuringElement.h"

//Hough
#include "itkImageRegionIterator.h"
#include "itkThresholdImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include <itkGradientMagnitudeImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <list>
#include "itkCastImageFilter.h"
#include "vnl/vnl_math.h"
#include "itkHoughTransform2DCirclesImageFilter.h"

// Min, max
#include <algorithm>
#include <vector>

// Coil Segmentation
#include <itkGrayscaleErodeImageFilter.h>
#include <itkGrayscaleDilateImageFilter.h>
#include <itkMeanImageFilter.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>

// Canny Levelset
#include "itkCannySegmentationLevelSetImageFilter.h"
#include "itkGradientAnisotropicDiffusionImageFilter.h"
#include "itkZeroCrossingImageFilter.h"

// Hole Filling
#include <itkBinaryFillholeImageFilter.h>

// Connected Components
#include "itkConnectedComponentImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"

// Resampling
#include "itkResampleImageFilter.h"
#include <itkNearestNeighborInterpolateImageFunction.h>
#include "itkScaleTransform.h"


// Needed by dcm2nii_dwii //
static bool StringLessThen( const std::string & s, const std::string & r )
{
	if ( s.size() > r.size() )
		return true;
	else
		return (s > r);
}
//----------------------------//



// Needed by dcm2nii_dwii ROI //
// ROI images have inverted direction. The ExtracImageFilter ignores any direction or origin change,
// i.e. origin and direction sycsessfully have been changed, but doesn't affect the extraction direction.
// So it's necessary to built the initial original ROI volume in an inverted way.
static bool StringMoreThen( const std::string & s, const std::string & r )
{
	if ( s.size() < r.size() )
		return true;
	else
		return (s < r);
}
//----------------------------//



int main( int argc, char* argv[] )
{

	if( argc < 2 )
	{
		std::cerr << std::endl << "This code implements a totally automated algorithm for "<<
			"MRI prostate volume segmentation." << std::endl;
		std::cerr << "This code is completely developed by the biomedical "
			<< "engineer Stefano Ravetto, committed by Polytechnic University of Turin"
			<< " and IRCCS (Institute for cancer research and threatment of Candiolo)."
			<< std::endl;
		std::cerr << "Sadly, this code in no more under development, though many"
		<< " improvements can be implemented." << std::endl;
		std::cerr << "For any information or if you are interested in further"
			<< " development of this code, please contact me at: "<< std::endl;
		std::cerr << "stefano.ravetto@studenti.polito.it" << std::endl;
		std::cerr << "https://it.linkedin.com/pub/stefano-ravetto/38/289/671" << std::endl << std::endl;

		std::cerr << "To run this code you need:" << std::endl;
		std::cerr << " · b0 DWI 2D slices of the volume of the prostate" << std::endl;
		std::cerr << " · T2 2D slices of the volume of the prostate" << std::endl;
		std::cerr << " · ROI boxes 2D slices of the volume of the prostate" << std::endl;
		std::cerr << "   (if you don't have the ROI the code can be easly fixed"
			<< " or the ROI easly obtained. See detailed description in the"
			<< " related part of the code)" << std::endl << std::endl;

		std::cerr << "The code expects the input images in the following way:" << std::endl;
		std::cerr << " · '~/reg_dwi/DWI1000/b0_reg_full', where inside must be stored"
			<< " the b0 DWI 2D images, in dicom format, and named as 'slice01.dcm,"
			<< " slice02.dcm, ...., slice10.dcm,...' " << std::endl;
		std::cerr << " · '~/T2/dcm', where inside must be stored"
			<< " the T2 2D images, in dicom format (no particular name is needed"
			<< " dicom information will be used instead" << std::endl;
		std::cerr << " · '~/reg_dwi/ROI_reg_full', where inside must be stored"
			<< " the ROI boxes 2D images, in dicom format, and named as 'slice01.dcm,"
			<< " slice02.dcm, ...., slice10.dcm,...' " << std::endl << std::endl;

		std::cerr << "The code expects as input the patient main directory,"
			<< " where inside its first sub-directories must be the three"
			<< " described above." << std::endl;
		std::cerr << "For example, you need to have a directory tree like:" << std::endl;
		std::cerr << "/patients/patientnumber1/reg_dwi/DWI1000/b0_reg_full" << std::endl;
		std::cerr << "/patients/patientnumber1/T2/dcm" << std::endl;
		std::cerr << "/patients/patientnumber1/reg_dwi/ROI_reg_full" << std::endl << std::endl;

		std::cerr << "'/patients/patientnumber1' is your core directory, in where"
			<< " all the other proccessing images and their directory will be created"
			<< " and saved." << std::endl << std::endl;

		std::cerr << "Please note that this is the first implementation of this code"
			<< " and the first method to try to be fully automated." << std::endl;
		std::cerr << "The code is written trying to be as general, multipurpose,"
		<< " and with less fixed value as possible, but this is not guaranteed." << std::endl;
		std::cerr << "I encourage you to read the code to evaluate that"
			<< " any of its parts and any fixed value fits your needs." << std::endl << std::endl;

		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0] << "	   CoreDirectory		[output mode]" << std::endl;
		std::cerr << "output mode is optional:" << std::endl;
		std::cerr << "0 (default): verbose mode (it's a very verbose mode,"
			<< " usefull to understand the code behavior for debug mode, or for" 
			<< " first time using)" << std::endl;
		std::cerr << "1          : short mode, with few informations and progress"
			<< " bars" << std::endl << std::endl;

		return EXIT_FAILURE;
	}

	// Output modality:
	// 0 (default): verbose mode (it's a very verbose mode, usefull to 
	// understand the code behavior for debug mode, or for first time using)
	// 1: short mode, with few informations and progress bars
	int output_mode = 0;
	
	if (argc == 3)
	{
	output_mode = atoi(argv[2]);
	}
	
	// Time probe "clock_all" to estimate the totale processing time, printed out
	// in the finale report
	itk::TimeProbe clock_all;
	
	// Time probe "clock" to estimate the crosscorretlation processing time,
	// used after in the code.
	itk::TimeProbe clock;

	// Here the clock starts
	clock_all.Start();


	// Now I create some of the folders in which a lot of output images are going
	// to be stored. Ohters folders will be create after in the code.
	
	// Here I create and obtain the main directory, it will be used often in the
	// code
	std::stringstream core_dir;
	// Setting the given input directory (in the form "patient/patientnumber")
	// to "core_dir"
	core_dir << argv[1];

	// Here I create a new "Segmented" directory, inside the core directory
	// Here is going to be saved the segmented images (2D, 3D, Chan & Vese, Canny)
	std::string segmented_dir;
	segmented_dir = core_dir.str();		// passing output directory to "segmented_dir"
	segmented_dir += "/Segmented";		// adding the new folder
	// Converting string to const char, required by "CreateDirectory"
	const char* char_segmented_dir = segmented_dir.c_str();
	// Creating the new "Segmented" directory
	itk::FileTools::CreateDirectory (char_segmented_dir);

	// Here I create a new "temporary" directory, inside the core directory,
	// to store some working files, that can be safetly deleted.
	std::string tmp_dir;
	tmp_dir = core_dir.str();		// passing output directory to "tmp_dir"
	tmp_dir += "/tmp";				// adding the new folder
	// Converting string to const char, required by "CreateDirectory"
	const char* char_tmp_dir = tmp_dir.c_str();
	// Creating the new "Segmented" directory
	itk::FileTools::CreateDirectory (char_tmp_dir);

	// Here I create a new "Cropped" directory, inside the core directory.
	// Here is going to be saved the cropped b0 images.
	std::string cropped_dir;
	cropped_dir = core_dir.str();		// passing output directory to "cropped_dir"
	cropped_dir += "/Cropped";			// adding the new folder
	// Converting string to const char, required by "CreateDirectory"
	const char* char_cropped_dir = cropped_dir.c_str();
	// Creating the new "cropped" directory
	itk::FileTools::CreateDirectory (char_cropped_dir);

	// Here I create a new "Post-processing" directory, inside the core directory.
	// here is going to be saved the post-processed images (CoilMask,
	// MaskedProstate, slice_pp, T2sub_slice).
	std::string postprocessing_dir;
	postprocessing_dir = core_dir.str();		// passing output directory to "cropped_dir"
	postprocessing_dir += "/Post-processing";	// adding the new folder
	// Converting string to const char, required by "CreateDirectory"
	const char* char_postprocessing_dir = postprocessing_dir.c_str();
	// Creating the new "cropped" directory
	itk::FileTools::CreateDirectory (char_postprocessing_dir);

	if (output_mode == 0)
	{
		std::cout << std::endl << "→ Initializing... [1/19]" << std::endl;
		std::cout << "     The main directory is: " << std::endl;
		std::cout << "     " << core_dir.str() << std::endl;
		std::cout << "     Creating the working directory: " << std::endl;;
		std::cout << "     " << segmented_dir << std::endl;
		std::cout << "     Creating the temporary directory: " << std::endl;;
		std::cout << "     " << tmp_dir << std::endl;
		std::cout << "     Creating the Cropped masked images directory: " << std::endl;;
		std::cout << "     " << cropped_dir << std::endl;
		std::cout << "     Creating the Post-processing images directory: " << std::endl;;
		std::cout << "     " << postprocessing_dir << std::endl;
		std::cout << std::endl;
	}



	/*=========================================================================
	 * 
	 dcm2nii_dwi - original b0 volume build
	 * 
	 =========================================================================*/
	// I proceed to build the voume from the original slices.
	// Since the dcm writing causes a lost of some dicom information,
	// a file name recognizion algorithm is used to  understand the slices order.
	// In order to work properly, slices file name must be like this:
	// slice01, slice02, ... , slice10, slice11, ...

	// Float type *nedeed* by the levelset filter
	typedef float    PixelType;			
	const unsigned int      Dimension = 3;

	typedef itk::OrientedImage< PixelType, Dimension > ImageType;
	typedef itk::ImageSeriesReader< ImageType > ReaderType;
	ReaderType::Pointer d2n_reader = ReaderType::New();

	typedef itk::GDCMImageIO       ImageIOType;
	ImageIOType::Pointer dicomIO = ImageIOType::New();

	d2n_reader->SetImageIO( dicomIO );

	typedef itk::GDCMSeriesFileNames NamesGeneratorType;
	NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

	d2n_reader->SetImageIO( dicomIO );

	nameGenerator->SetUseSeriesDetails( false );

	// The orignal b0 images *must* be placed in a directory named:
	// ".../reg_dwi/DWI1000/b0_reg_full"
	// where "..." is the core directory (core_dir).
	std::string b0_dir;
	// Passing input directory to "b0_dir"
	b0_dir = core_dir.str();
	// Completing b0_dir with the full directory
	b0_dir += "/reg_dwi/DWI1000/b0_reg_full";	

	nameGenerator->SetDirectory( b0_dir );

	if (output_mode == 0)
	{
		std::cout << std::endl << "→ Analizing directory... [2/19]" << std::endl;
		std::cout << "     The directory: " << std::endl;
		std::cout << "     " << b0_dir << std::endl;
		std::cout << "     Contains the following DICOM Series: ";
		std::cout << std::endl;
	}

	typedef std::vector< std::string >    SeriesIdContainer;
	typedef std::vector< std::string >   FileNamesContainer;
	FileNamesContainer fileNames;

	const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

	SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
	SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
	while( seriesItr != seriesEnd )
	{
		if (output_mode == 0)
		{
			std::cout << "     " << seriesItr->c_str() << std::endl;
		}
		fileNames.insert(fileNames.end(), nameGenerator->GetFileNames(seriesItr->c_str()).begin(), nameGenerator->GetFileNames(seriesItr->c_str()).end());
		seriesItr++;
	}

	std::sort(fileNames.begin(), fileNames.end(), StringLessThen);

	d2n_reader->SetFileNames( fileNames );

	try
	{
		d2n_reader->Update();
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}

	typedef itk::OrientImageFilter<ImageType,ImageType> OrienterType;
	OrienterType::Pointer orienter = OrienterType::New();

	orienter->UseImageDirectionOn();
	orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
	orienter->SetInput(d2n_reader->GetOutput());

	typedef itk::ImageFileWriter< ImageType > WriterType;
	WriterType::Pointer writer = WriterType::New();

	// Generating volume output file name, usign the output directory given.
	// The output volume is going to be in nifti format
	std::string tmp_dir_volume;
	std::stringstream out_dir;
	out_dir << argv[1];				// setting the given output directory
	tmp_dir_volume = out_dir.str();		// passing output directory to "tmp_dir_volume"
	tmp_dir_volume += "/Volume_Original.nii";		// post-adding "/---.nii"

	writer->SetFileName( tmp_dir_volume );

	writer->SetInput( orienter->GetOutput() );

	if (output_mode == 0)
	{
		std::cout << std::endl << std::endl << "→ Building volume  [3/19]" << std::endl;
		std::cout << std::endl << "     Writing the original volume image as: " << std::endl;
		std::cout << "     " << tmp_dir_volume << std::endl;
	}
	
	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}
    


	/*=========================================================================
	 * 
	 Extracting information part
	 * 
	 =========================================================================*/
	// Here I extract some essential information from the original builted b0 volume.
	// All the next slices/volumes must have this parameter for consistency!


	// Now I get the original spacing i.e. pixel resolution of the original image,
	// taken from the rebuilt volume. It can't be taken from the original slices
	// because the initial DICOM series recongnizer generates vector (non-scalar)
	// image type.
	typedef itk::Image< PixelType, 2 > SpacingImageType;
	typedef itk::ImageFileReader< SpacingImageType > SpacingReaderType;
	SpacingReaderType::Pointer Spacing_reader = SpacingReaderType::New();
	Spacing_reader->SetFileName( tmp_dir_volume );
	Spacing_reader->Update();
	SpacingImageType::ConstPointer SpacingImage = Spacing_reader->GetOutput();
	SpacingImageType::SpacingType spacing = SpacingImage->GetSpacing();
	if (output_mode == 0)
	{
		std::cout << std::endl << "     Original spacing (pixel resolution) is: " << spacing << std::endl;
	}
  
  /* // This part is commented because even the direction is changed,
     // the ExtractImageFIlter ignores it, beginnging extraction from the wrong
	 // slice.
     // On original image this is not a issue, because the direction is corretc,
     // and this is only a control, but on ROI images the direction is wrong.
     // So the only other way is to rebuil the ROI volume in a reverse way.
  // Now I get the original direction of the original image, taken from the rebuilt volume
  // It can't be taken from the original slices becouse the initial DICOM series recongnizer generates vector (non-scalar) image type
  typedef itk::Image< PixelType, 3 > DirectionImageType;
  typedef itk::ImageFileReader< DirectionImageType > DirectionReaderType;
  DirectionReaderType::Pointer Direction_reader = DirectionReaderType::New();
  Direction_reader->SetFileName( tmp_dir_volume );
  Direction_reader->Update();
  DirectionImageType::ConstPointer DirectionImage = Direction_reader->GetOutput();
  DirectionImageType::DirectionType direction = DirectionImage->GetDirection();
  std::cout << std::endl << "     Original direction is: "<< std::endl << direction << std::endl;
  */



	/*=========================================================================
	 * 
	 Gaussian Discrete 3D Filter
	 * 
	 =========================================================================*/
	// Here I perform a light smoothing on the volume
	// Previously generated original volume will be overwritted, because is no
	// longer needed.

	try
	{
		if (output_mode == 0)
		{
			std::cout << std::endl << "→ Now 3D Gaussian Smoothing Filter is going to be performed ...  [4/19]"<< std::endl;
		}

		// Setup types
		typedef itk::Image< float, 3 > UnsignedCharImageType;
		typedef itk::Image< float, 3 >         FloatImageType;

		typedef itk::ImageFileReader< UnsignedCharImageType >  readerType;
		typedef itk::ImageFileWriter< FloatImageType >  writerType;

		typedef itk::DiscreteGaussianImageFilter<
			UnsignedCharImageType, FloatImageType >  filterType;

		// Create and setup a reader
		readerType::Pointer reader = readerType::New();
		writerType::Pointer writer = writerType::New();
		reader->SetFileName(tmp_dir_volume);
		reader->Update();

		if (output_mode == 0)
		{
			std::cout << "     Loaded Original Volume: "<< tmp_dir_volume << std::endl;
		}

		// Create and setup a Gaussian filter
		double variance = 2.0;
		filterType::Pointer gaussianFilter = filterType::New();
		gaussianFilter->SetInput( reader->GetOutput() );
		gaussianFilter->SetVariance(variance);

		writer->SetInput(gaussianFilter->GetOutput() );
		writer->SetFileName( tmp_dir_volume );
		writer->Update();
		if (output_mode == 0)
		{
			std::cout << "     Overwrited Original Volume with Smoothed one: "<< tmp_dir_volume << std::endl;
			std::cout << "     Done!"<< std::endl;
		}
	}
	catch(itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}
	


	/*=========================================================================
	 
	 dcm2nii_dwi - ROI volume build
	 
	 =========================================================================*/
	// Now I proceed to build the ROI volume from the slices,
	// in the same way done for b0.
	// ROIs are 2d boxes masks, that aproximatively contatains the prostate, for
	// each z height. This masks are usefull to elimate sone non prostatic things
	// from the images (noise).
	// This masks are not indispensable, bute very usefull.
	// If not provided, they can be easily created using Hough transform
	// to find the prostate center, and x and y measures of the boxes are
	// generated using literature informations of the patologic prostates sizes.

	ReaderType::Pointer ROId2n_reader = ReaderType::New();
	ImageIOType::Pointer ROIdicomIO = ImageIOType::New();
	ROId2n_reader->SetImageIO( ROIdicomIO );

	NamesGeneratorType::Pointer ROInameGenerator = NamesGeneratorType::New();

	ROId2n_reader->SetImageIO( ROIdicomIO );

	ROInameGenerator->SetUseSeriesDetails( false );

	// The ROI images *must* be placed in a directory named:
	// ".../reg_dwi/ROI_reg_full"
	// where "..." is the core directory (core_dir).
	std::string ROI_dir;
	ROI_dir = core_dir.str();				// passing input directory to "ROI_dir"
	ROI_dir += "/reg_dwi/ROI_reg_full";		// completing ROI_dir with the full directory
	
	ROInameGenerator->SetDirectory( ROI_dir );

	if (output_mode == 0)
	{
		std::cout << std::endl << "→ Analizing directory...  [5/19]" << std::endl;
		std::cout << "     The directory: " << std::endl;
		std::cout << "     " << ROI_dir << std::endl;
		std::cout << "     Contains the following DICOM Series: ";
		std::cout << std::endl;
	}

	typedef std::vector< std::string >    ROISeriesIdContainer;
	typedef std::vector< std::string >   ROIFileNamesContainer;

	ROIFileNamesContainer ROIfileNames;

	const ROISeriesIdContainer & ROIseriesUID = ROInameGenerator->GetSeriesUIDs();

	ROISeriesIdContainer::const_iterator ROIseriesItr = ROIseriesUID.begin();
	ROISeriesIdContainer::const_iterator ROIseriesEnd = ROIseriesUID.end();
	while( ROIseriesItr != ROIseriesEnd )
	{
		if (output_mode == 0)
		{
			std::cout << "     " << ROIseriesItr->c_str() << std::endl;
		}
			ROIfileNames.insert(ROIfileNames.end(), ROInameGenerator->
			                    GetFileNames(ROIseriesItr->c_str()).begin(),
			                    ROInameGenerator->GetFileNames(ROIseriesItr->c_str()).end());
			ROIseriesItr++;
		
	}

	std::sort(ROIfileNames.begin(), ROIfileNames.end(), StringMoreThen);

	ROId2n_reader->SetFileNames( ROIfileNames );

	try
	{
		ROId2n_reader->Update();
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}

	OrienterType::Pointer ROIorienter = OrienterType::New();

	ROIorienter->UseImageDirectionOn();
	ROIorienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
	ROIorienter->SetInput(ROId2n_reader->GetOutput());

	WriterType::Pointer ROIwriter = WriterType::New();

	// Generating volume output file name, usign the output directory given
	std::string ROItmp_dir_volume;
	ROItmp_dir_volume = out_dir.str();		// passing output directory to "tmp_dir_volume"
	ROItmp_dir_volume += "/ROIVolume_Original.nii";		// post-adding "/---.nii"

	ROIwriter->SetFileName( ROItmp_dir_volume );
	ROIwriter->SetInput( ROIorienter->GetOutput() );

	if (output_mode == 0)
	{
		std::cout << std::endl << std::endl << "→ Building volume  [6/19]" << std::endl;
		std::cout << std::endl << "     Writing the original ROI volume image as: " << std::endl;
		std::cout << "     " << ROItmp_dir_volume << std::endl;
	}

	try
	{
		ROIwriter->Update();
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}



	/*=========================================================================
	 
	 Change ROI Z direction 
	 
	 =========================================================================*/
	// This part is commented because even if the z direction is inverted,
	// this doesn't affect the direction of the builted volume.
	// As described before, the only way semms to be to built the volume
	// in a reverse order.
	// This part remains here because can be useful in future imkplementations.
	/*
	 typedef itk::ChangeInformationImageFilter< ImageType > ChDirection_FilterType;
	 ChDirection_FilterType::Pointer ChDirection_filter = ChDirection_FilterType::New();


	 // This part is commented becouse even the direction is changed,
	 // the ExtractImageFilter ignores it, beginnging extraction from the wrong slice.
	 // On ROI images the direction is wrong.
	 // So the only other way is to rebuil the ROI volume in a reverse way.
	 DirectionReaderType::Pointer ROIDirection_reader = DirectionReaderType::New();
	 ROIDirection_reader->SetFileName( ROItmp_dir_volume );
	 ROIDirection_reader->Update();
	 DirectionImageType::ConstPointer ROIDirectionImage = ROIDirection_reader->GetOutput();
	 DirectionImageType::DirectionType ROIdirection = ROIDirectionImage->GetDirection();
	 std::cout << std::endl << "     ROI Z direction is: "<< ROIdirection(2,2) << std::endl;

	 // ROI Z direction must be inverted (ad-hoc fix)
	 ROIdirection(2,2) = 1;

	 std::cout << "     → Fixing direction" << std::endl;
	 ChDirection_filter->SetOutputDirection( ROIdirection );
	 ChDirection_filter->ChangeDirectionOn();
	 std::cout << "          New correct Z direction is: "<< ROIdirection << std::endl;
	 std::cout << "          Fixed!"<< std::endl<< std::endl;




	 // Even changin origin z origin doesn't work...

	 DirectionReaderType::Pointer ROIOrigin_reader = DirectionReaderType::New();
	 ROIOrigin_reader->SetFileName( ROItmp_dir_volume );
	 ROIOrigin_reader->Update();
	 DirectionImageType::ConstPointer ROIOriginImage = ROIOrigin_reader->GetOutput();
	 DirectionImageType::PointType ROIorigin = ROIOriginImage->GetOrigin();
	 std::cout << std::endl << "     ROI Z origin is: "<< ROIorigin[2] << std::endl;

	 // ROI Z Origin must be inverted (ad-hoc fix)
	 ROIorigin[2] = 23;

	 std::cout << "     → Fixing Origin" << std::endl;
	 ChDirection_filter->SetOutputOrigin( ROIorigin );
	 ChDirection_filter->ChangeOriginOn();
	 std::cout << "          New correct Z Origin is: "<< ROIorigin << std::endl;
	 std::cout << "          Fixed!"<< std::endl<< std::endl;

	 ChDirection_filter->SetInput( ROIorienter->GetOutput() );
	 ChDirection_filter->UpdateLargestPossibleRegion();

	 WriterType::Pointer ChDirection_ROIwriter = WriterType::New();
	 ChDirection_ROIwriter->SetFileName( ROItmp_dir_volume );

	 ChDirection_ROIwriter->SetInput( ChDirection_filter->GetOutput() );
	 try
	 {
		 ChDirection_ROIwriter->Update();
	}
		catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}
	*/




	/*=========================================================================
	  
	 ExtractImageWriter
	  
	 =========================================================================*/
	// Here b0 smoothed volume and ROI volume are processed in order to
	// extract them slices one by one, and to perform some pre-processing
	// operations:
	// 1 - masking smoothed b0 with the ROI box
	// 2 - maxiimum intensity value, that will be uses later in the code
	//	   (H automatic search algorithm)
	// 3 -Origin and spacing control and fix
	// Moreover, the crosscorelation between all the slices of the volume is
	// computed, ad standard deviation and mean intensity vlaues are extrapolated.
	// This parameters is going to be used later in the code, in the 
	// Non-Prostatic Exclusion Criteria

	// Float type *nedeed* by tha levelset filter
	typedef float InputPixelType;
	typedef float OutputPixelType;
	typedef itk::Image< InputPixelType, 3 > InputImageType;
	typedef itk::Image< OutputPixelType, 2 > OutputImageType;

	// Declaration of readed and Writer Type
	//	Original
	ReaderType::Pointer ExtIm_reader = ReaderType::New();
	WriterType::Pointer ExtIm_writer = WriterType::New();
	//	ROI
	ReaderType::Pointer ROIExtIm_reader = ReaderType::New();
	WriterType::Pointer ROIExtIm_writer = WriterType::New();

	// Here I get the volume where the slices must be extracted
	// 	Original
	ExtIm_reader->SetFileName( tmp_dir_volume );
	//	ROI
	ROIExtIm_reader->SetFileName( ROItmp_dir_volume );

	// Defining the Extraction Filter
	typedef itk::ExtractImageFilter< InputImageType,
	OutputImageType > FilterType;

	// Creating Pointer
	//	Original
	FilterType::Pointer ExtIm_filter = FilterType::New();
	ExtIm_filter->InPlaceOn();
	ExtIm_filter->SetDirectionCollapseToSubmatrix();
	//	ROI
	FilterType::Pointer ROIExtIm_filter = FilterType::New();
	ROIExtIm_filter->InPlaceOn();
	ROIExtIm_filter->SetDirectionCollapseToSubmatrix();

	// The ExtractImageFilter requires a region to be defined by the
	// user. The region is specified by an \doxygen{Index} indicating the
	// pixel where the region starts and an \doxygen{Size} indication how many
	// pixels the region has along each dimension. In order to extract a $2D$
	// image from a $3D$ data set, it is enough to set the size of the region
	// to $0$ in one dimension. This will indicate to
	// ExtractImageFilter that a dimensional reduction has been
	// specified. Here I take the region from the largest possible region of
	// the input image. Note that UpdateOutputInformation() ** (view NOTE) is being
	// called first on the reader, this method updates the meta-data in
	// the outputImage without actually reading in the bulk-data.

	//	Original
	ExtIm_reader->Update();		// NOTE: do not use UpdateOutputInformation(),
	// becouse in this code is used a loop to process al the images,
	//and in this way some input image informations is not updated at every cycle.
	//Use Update() instead.
	InputImageType::RegionType inputRegion =
		ExtIm_reader->GetOutput()->GetLargestPossibleRegion();
	//	ROI
	ROIExtIm_reader->Update();		// NOTE: do not use UpdateOutputInformation(),
	// becouse in this code is used a loop to process al the images,
	//and in this way some input image informations is not updated at every cycle.
	//Use Update() instead.
	InputImageType::RegionType ROIinputRegion =
		ROIExtIm_reader->GetOutput()->GetLargestPossibleRegion();

	// I take the size from the region and I collapse the size in the $Z$
	// component by setting its value to $0$. This will indicate to the
	// ExtractImageFilter that the output image should have a
	// dimension less than the input image.

	InputImageType::SizeType size = inputRegion.GetSize();
	long unsigned int size_z = size[2];		// size_z = total slices number

	size[2] = 0;					// size[z]

	// Note that in this case I are extracting a $Z$ slice, and for that
	// reason, the dimension to be collapsed in the one with index $2$. You
	// may keep in mind the association of index components
	// $\{X=0,Y=1,Z=2\}$. If I am interested in extracting a slice
	// perpendicular to the $Y$ axis I would have set \code{size[1]=0;}.

	// Then, I take the index from the region and set its $Z$ value to the
	// slice number I want to extract. In this example I obtain the slice
	// number from the command line arguments.

	InputImageType::IndexType start = inputRegion.GetIndex();
	long unsigned int sliceNumber = 0; 		// Beginning extraction slice //I begin from slice 0 to maximun z slice i.e. z_slice
	if (output_mode == 0)
	{
		std::cerr <<"     Total numer of slices is: "<< size_z << std::endl;
	}

	long unsigned int sliceNumberShowed;		// Here I declare the slice number printed out and used to generate slices (see following description)

	// Here i instantiate a vector containing the maximum intensity value
	// of each b0 slice. This is going to be used later in the Chan and Vese
	// automatic paramenters search.
	std::vector<float> maximum_intensity_b0 (size_z, 0);

	std::cout << std::endl <<"→ Now pre-processing controls and filters are going to be performed ...  [7/19]"<< std::endl;

	

//------------------------ EXTRACT SLICES FOR STARTS ------------------------//

  for ( sliceNumber; sliceNumber < size_z; sliceNumber++ )

    {		// for cycle open

		// Here I declare the printed slice number "sliceNumberShowed", where
		// its range is [1, size_z].
		// Instead, "sliceNumber" is the real slice number used by the algorithm,
		// its range is [0, size_z - 1].
		sliceNumberShowed = sliceNumber + 1;

		// Using slicenUmberShowed instead of sliceNumber produces a more human
		// readeable output and gives coerency with slices generated file names
		// with other IRCCS study (ad-hoc fix).
		if (output_mode == 0)
		{
			std::cerr <<"     → Now processing slice "<< sliceNumberShowed << std::endl
				<< std::endl;
		}

		start[2] = sliceNumber;				// start[z]

		// Finally, an \doxygen{ImageRegion} object is created and initialized with
		// the start and size I just prepared using the slice information.

		InputImageType::RegionType desiredRegion;
		desiredRegion.SetSize( size );
		desiredRegion.SetIndex( start );

		// Then the region is passed to the filter using the
		// SetExtractionRegion() method.

		//		Original
		ExtIm_filter->SetExtractionRegion( desiredRegion );
		ExtIm_filter->SetInput( ExtIm_reader->GetOutput() );
		//		ROI
		ROIExtIm_filter->SetExtractionRegion( desiredRegion );
		ROIExtIm_filter->SetInput( ROIExtIm_reader->GetOutput() );

		//  I now define the image type using a particular pixel type and
		//  dimension. In this case the \code{float} type is used for the pixels
		//  due to the requirements of the smoothing filter.

		const unsigned int Dimension = 2;
		typedef float ScalarPixelType;
		typedef itk::Image< ScalarPixelType, Dimension > InternalImageType;

		// inputImage is the extracted region
		//		Original
		InternalImageType::Pointer inputImage = ExtIm_filter->GetOutput();
		//		ROI
		InternalImageType::Pointer ROIinputImage = ROIExtIm_filter->GetOutput();

		// Here I get the maximum intensity value of the b0 MRI volume,
		// used to define the histogram upper value and its bins
		typedef itk::StatisticsImageFilter<OutputImageType> H_StatisticsImageFilterType;
		H_StatisticsImageFilterType::Pointer H_statisticsImageFilter
			= H_StatisticsImageFilterType::New ();
		H_statisticsImageFilter->SetInput(inputImage);
		H_statisticsImageFilter->UpdateLargestPossibleRegion();
		maximum_intensity_b0 [sliceNumber] = H_statisticsImageFilter->GetMaximum();
		if (output_mode == 0)
		{
			std::cout << std::endl << "               Maximum intensity value is: " 
				<< maximum_intensity_b0 [sliceNumber] << std::endl;
		}



		/*======================================================================
		 
		 ChangeInformationImageFilter
		 
		 =====================================================================*/
		// Now some control precesses are needed:
		// Origin must be [0,0,0] (to avoid "requested region out of laregest possible region" issue)
		// Spacing (aka PixelResolutin) must be equal to the original (ITK fails auto-recognizment in certain cases and sests a wrons Spacing)
		// Z direction in needed to ensur all processed slices of different volume progresses in the same direction

		InternalImageType::PointType iorigin = inputImage->GetOrigin();
		if (output_mode == 0)
		{
			std::cout << "          → Image origin control now is taking place..." << std::endl;
			std::cout << "               input image origin: " << iorigin << std::endl;
		}

		typedef itk::ChangeInformationImageFilter< InternalImageType > ChInfo_FilterType;
		ChInfo_FilterType::Pointer ChInfo_filter = ChInfo_FilterType::New();
		InternalImageType::SpacingType ext_spacing = inputImage->GetSpacing();

		// Fixing origin to (0,0,0) to avoid "out of largest possible region" error.
		InternalImageType::PointType origin = 0;
		if (output_mode == 0)
		{
			std::cout << "               applying new origin: " << origin << std::endl;
		}
		ChInfo_filter->SetOutputOrigin( origin );	
		ChInfo_filter->ChangeOriginOn();
		ChInfo_filter->SetInput( inputImage );

		InternalImageType::Pointer featureImage = ChInfo_filter->GetOutput();
		InternalImageType::PointType forigin = featureImage->GetOrigin();
		if (output_mode == 0)
		{
			std::cout << "               output image origin: " << forigin << std::endl;
			std::cout << "               Done!" << std::endl;
			std::cout << " " << std::endl;
		}

		// Fixing Spacing i.e. Pixel resolution issue. The ExtractImageFilter automatic changes the original spacing to (1,1),
		// but Pixel resolution as a phisical meaning!
		// Original Pixel Resolution is needed to make Level Set filter work properly.
		if (output_mode == 0)
		{
			std::cout << "          → Fixing spacing (pixel resolution)" << std::endl;
		}
		ChInfo_filter->SetOutputSpacing( spacing );
		ChInfo_filter->ChangeSpacingOn();
		if (output_mode == 0)
		{
			std::cout << "               Wrong spacing is: "<< ext_spacing << std::endl;
			std::cout << "               New correct spacing is: "<< spacing << std::endl;
			std::cout << "               Fixed!"<< std::endl<< std::endl;
		}

		ChInfo_filter->UpdateLargestPossibleRegion();



		/*======================================================================
		  
		 Masking part
		  
		 =====================================================================*/
		// Here I perform the masking operations using the ROI extracted slices 
		// over the smooted b0 extracted slices.

		if (output_mode == 0)
		{
			std::cout << "          → Masking filter operations is going to be performed ..." <<  std::endl;
			std::cout << "               Generating masked (aka 'Cropped') and inverse-masked (aka 'Filled') images ... Done!   " << std::endl;
		}

		// MaskImageFilter masks the 0 pixel (as expected). So I can take the internal ROI region.
		typedef itk::MaskImageFilter< InternalImageType, InternalImageType > MaskFilterType;
		MaskFilterType::Pointer maskFilter = MaskFilterType::New();
		maskFilter->SetInput(featureImage);
		maskFilter->SetMaskImage(ROIinputImage);

		InternalImageType::Pointer CroppedImage = maskFilter->GetOutput();

		// I instantiate reader and writer types in the following lines.
		typedef itk::ImageFileReader< InternalImageType > ReaderType;
		ReaderType::Pointer CnV_reader = ReaderType::New();

		// I need to save any slice in DCM format. Levelset filter needs floating
		// type image, but is not supported in DCM (and GDCM), so a *casting* 
		// operation is needed.
		// Here I define the conversion format -short- and its writer
		typedef short shortScalarPixelType;
		typedef itk::Image< shortScalarPixelType, Dimension > shortInternalImageType;
		typedef itk::ImageFileWriter< shortInternalImageType > shortWriterType;
		shortWriterType::Pointer CnV_writer = shortWriterType::New();

		// Generating single slice output file name, usign the output directory given.
		// I use the cropped image becouse in this way I leave out non usefull region.

		// Generating actual slice numer string
		std::string sn;
		std::stringstream out_slice;
		out_slice << sliceNumberShowed;
		sn = out_slice.str();		// actual slice number passed to "sn"

		// Adding other parts of file name, 
		// so I have: tmp_dir_slice = tmp_dir + "/slice" + s + ".nii"
		std::string tmp_dir_slice;
		tmp_dir_slice += cropped_dir;
		tmp_dir_slice += "/slice_cropped";
		if ( sliceNumberShowed < 10 )
		{
			// I want a name generation like this:
			// slice01, slice02, ... , slice10, slice11, ...
			tmp_dir_slice += "0";
		}
		tmp_dir_slice += sn;
		// Output slice format. dcm requires a casting operation, because float
		// is not supported.
		tmp_dir_slice += ".dcm";

		if (output_mode == 0)
		{
			std::cout << "               Cropped image saved as: " << tmp_dir_slice << std::endl;
		}

		CnV_writer->SetFileName( tmp_dir_slice );

		try
		{
			// I need to save any slice in DCM format. Levelset filter needs 
			// floating type image, but is not supported in DCM (and GDCM),
			// so a *casting* operation is needed.
			// Here casting operation takes palce, from float (levelset filter)
			// to short (dcm).

			typedef itk::Image<float, 2>  FloatImageType;
			typedef itk::Image<short, 2>  ShortImageType;
			typedef itk::CastImageFilter< FloatImageType, ShortImageType > CastFilterType;
			CastFilterType::Pointer castFilter = CastFilterType::New();
			castFilter->SetInput(CroppedImage);

			CnV_writer->SetInput( castFilter->GetOutput() );
			CnV_writer->Update();

			if (output_mode == 0)
			{
				std::cout << "               Done!" << std::endl;
				std::cout << " " << std::endl;
			}
		}
		catch( itk::ExceptionObject & excep )
		{
			std::cerr << "Exception caught !" << std::endl;
			std::cerr << excep << std::endl;
			return -1;
		} 

	}		// main for cycle close
	//---------------------- EXTRACT SLICES FOR ENDS ----------------------//




	/*=========================================================================
	  
	 Non-prostatic Slice Exclusion Criteria
	 
	 =========================================================================*/
	// Here I proceed to try to find when the slices contains no more prostate.
	// To achieve this, I use the crosscorrelation to mache the central slice,
	// which has the best prostate image, with the all other slices in the volume.
	// Than I obtain the standard deviation from any comparison and I perform
	// some operations fully described afterwards.

	if (output_mode == 0)
	{
		std::cout << "→ Now Crosscorrelation analysis between all the slices is going to be performed ...  [8/19]" << std::endl << std::endl;
	}

	// This is the core slice to be compared with the other slices.
	// I set it as the central slice.
	const int central_slice = size_z / 2;

	// Since crosscorrelation algorithm is very time-consuming, I don't
	// process all the slices, but instead I set a sampling interval.
	const int sample_interval = 1;

	// Here I save the obtained standard deviation from all the comparison we
	// will do.
	std::vector<float> sd (size_z, 0);
	// Here I save the obtained mean intensity from all the comparison we
	// will do.
	std::vector<float> meani (size_z, 0);

	// Here I read the central slice from the origianl b0 slices (with no mask
	// or filtering)

	// Generating actual slice numer string
	std::string sn;
	std::stringstream out_slice;
	out_slice << central_slice;
	sn = out_slice.str();		// actual slice number passed to "sn"

	// Generating the file names to be read, so I kane keep the cropped slices
	// from the cropped folder.
	std::string b0_begin_dir_slice;
	b0_begin_dir_slice += b0_dir;
	b0_begin_dir_slice += "/slice";

	if ( central_slice < 10 )
	{

		// I want a name generation like this:
		// slice01, slice02, ... , slice10, slice11, ...
		b0_begin_dir_slice += "0";
	}
	b0_begin_dir_slice += sn;
	// Output slice format. dcm requires a casting operation, because float is
	// not supported.
	b0_begin_dir_slice += ".dcm";

	// Here I instantiate the vector tha is going to contain the homogeneity
	// (better said "dishomogeneity") values.
	std::vector<float> homogeneityValues (0 , 0);

	// Here I set a loop cycle to perform the comparison between our core slice
	// and the other choosed slices.

	//---------------------- CROSSCORRELATION FOR START ----------------------//
	for ( int k = 1; k <= size_z; k += sample_interval)
	{
		// Clock probe for the crosscorrelation starts here. This will estimate
		// the crosscorrelation time to complete.
		clock.Start();

		// Avoiding to exceed the maximum slice number in the volume,
		// due to the sampling applied
		if (k > size_z)
		{
			break;
		}

		// Reading one by one the choosed original slices.

		// Generating actual slice numer string
		std::string sn;
		std::stringstream out_slice;
		out_slice << k;
		sn = out_slice.str();		// actual slice number passed to "sn"

		// Generating the file names to be read, so I can keep the cropped slices
		// from the cropped folder.
		std::string b0_dir_slice;
		b0_dir_slice += b0_dir;
		b0_dir_slice += "/slice";

		if ( k < 10 )
		{
			// I want a name generation like this:
			// slice01, slice02, ... , slice10, slice11, ...
			b0_dir_slice += "0";
		}
		b0_dir_slice += sn;
		// Output slice format. dcm requires a casting operation, because float
		// is not supported.
		b0_dir_slice += ".dcm";

		// Now the Crosscorrelation follows.

		typedef itk::Image<float, 2> FloatImageType;
		typedef itk::Image<unsigned char, 2> UnsignedCharImageType;

		typedef itk::ImageFileReader<FloatImageType> ReaderType;

		// Read the images
		// reader1 and cross1 is the core image (passed to the kernel)
		ReaderType::Pointer reader1 = ReaderType::New();
		reader1->SetFileName(b0_begin_dir_slice);
		reader1->Update();
		// reader2 and cross2 are the other images
		ReaderType::Pointer reader2 = ReaderType::New();
		reader2->SetFileName(b0_dir_slice);
		reader2->Update();

		if (output_mode == 0)
		{
			std::cout << "     → Comparing slice " << central_slice << " with slice " 
				<< k << std::endl;
		}

		// Since crosscorrelation is very time-consuming, I reduce
		// the slices size in order to speed-up the whole process.
		// I pay a very littel price in term of precision.

		typedef itk::ShrinkImageFilter <FloatImageType, FloatImageType>
			ShrinkImageFilterType;

		ShrinkImageFilterType::Pointer shrinkFilter1
			= ShrinkImageFilterType::New();
		shrinkFilter1->SetInput(reader1->GetOutput());

		// Shrink the first dimension by a factor of 4 
		shrinkFilter1->SetShrinkFactor(0, 4);

		// Shrink the second dimension by a factor of 4
		shrinkFilter1->SetShrinkFactor(1, 4);

		FloatImageType::Pointer cross1 = shrinkFilter1->GetOutput();

		ShrinkImageFilterType::Pointer shrinkFilter2
			= ShrinkImageFilterType::New();
		shrinkFilter2->SetInput(reader2->GetOutput());

		// Shrink the first dimension by a factor of 4 
		shrinkFilter2->SetShrinkFactor(0, 4);

		// Shrink the second dimension by a factor of 4
		shrinkFilter2->SetShrinkFactor(1, 4);

		FloatImageType::Pointer cross2 = shrinkFilter2->GetOutput();

		shrinkFilter1->Update();
		shrinkFilter2->Update();

		if (output_mode == 0)
		{
			std::cout << "          Slice " << central_slice << " temporary resized to " 
				<< cross1->GetLargestPossibleRegion().GetSize() << std::endl;
			std::cout << "          Slice " << k << " temporary resized to " 
				<< cross2->GetLargestPossibleRegion().GetSize() << std::endl;
		}
		

		// Extract a small region
		typedef itk::RegionOfInterestImageFilter< FloatImageType,
		FloatImageType > ExtractFilterType;

		ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();

		FloatImageType::IndexType start;
		start.Fill(0);

		// The kernel need an odd dimensions image.
		// So I must ensure this.

		// new_size is the shrinked size
		FloatImageType::SizeType new_size = cross2->GetLargestPossibleRegion().GetSize();

		FloatImageType::SizeType patchSize;
		int odd_size = new_size[0] / 2 * 2 - 1;
		patchSize.Fill(odd_size);

		FloatImageType::RegionType desiredRegion(start,patchSize);

		extractFilter->SetRegionOfInterest(desiredRegion);
		extractFilter->SetInput(cross1);
		extractFilter->Update();

		// Perform normalized correlation
		// <input type, mask type (not used), output type>
		typedef itk::NormalizedCorrelationImageFilter<FloatImageType, FloatImageType, FloatImageType> CorrelationFilterType;

		itk::ImageKernelOperator<float> kernelOperator;
		//kernelOperator.SetImageKernel(extractFilter->GetOutput());
		kernelOperator.SetImageKernel(extractFilter->GetOutput());

		// The radius of the kernel must be the radius of the patch, NOT the size of the patch
		itk::Size<2> radius = extractFilter->GetOutput()->GetLargestPossibleRegion().GetSize();
		radius[0] = (radius[0]-1) / 2;
		radius[1] = (radius[1]-1) / 2;

		kernelOperator.CreateToRadius(radius);

		CorrelationFilterType::Pointer correlationFilter = CorrelationFilterType::New();

		correlationFilter->SetInput(cross2);
		correlationFilter->SetTemplate(kernelOperator);
		correlationFilter->Update();
		
		typedef itk::RescaleIntensityImageFilter< FloatImageType, FloatImageType > RescaleFilterType;
		RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
		rescaleFilter->SetInput(correlationFilter->GetOutput());
		rescaleFilter->SetOutputMinimum(0);
		rescaleFilter->SetOutputMaximum(255);
		rescaleFilter->Update();

		// Here I get the standard deviation and the mean of the crosscorrelation
		// images
		typedef itk::StatisticsImageFilter<FloatImageType> StatisticsImageFilterType;
		StatisticsImageFilterType::Pointer statisticsImageFilter
			= StatisticsImageFilterType::New ();
		statisticsImageFilter->SetInput(rescaleFilter->GetOutput());
		statisticsImageFilter->Update();

		sd [k-1] = statisticsImageFilter->GetSigma();
		meani [k-1] = statisticsImageFilter->GetMean();

		if (output_mode == 0)
		{
			std::cout << "          The Standard Deviation is: " << sd [k-1]  << std::endl;
			std::cout << "          The mean Intensity is: " << meani [k-1]  << std::endl;
			std::cout << "          Done!" << std::endl;
		}

		// --- -- - Cross-correlation part ends here - -- --- //



		/*======================================================================
		 
		 Dishomogeneity
		 
		 =====================================================================*/

		// Here I use the intensity information of the crosscorrelation images
		// to extrapolate a dishomegeneity coefficient value for each
		// crosscorrelation image.

		// To achieve this I count the number of each pixel with a certain
		// intensity value (like an histogram of intensity).

		// Current pixel index I am scanning
		OutputImageType::IndexType currentIndex;
		// Histogram bins (in wich I divide the intensity range)
		int histogramBins = 50;
		// Number of effective intensity levels ("+1" to take in account of
		// the pixel values out of range of the last 50th bin)
		int levelsNumber = histogramBins + 1;
		// The pixel range of each bins (255 is the maximum intensity value
		// set via the rescal filter before)
		float intensityIncrement = 255 / histogramBins;
		// A vector to store the cont of each pixel intensity, in every level
		std::vector<float> intensityCount (levelsNumber , 0);
		

		// Here I scan every row and every column and I classify each pixel for
		// each cycle (row and col remains fixed until the pixel is classified)
		for (int row = 0; row <= new_size[0]; row++)
		{

			for (int col = 0; col <= new_size[1]; col++)
			{

				// Every new cycle I must redefine:
				// Flag set to pixel being not classified
				bool pixel_classified = false;
				// Actual (first) level minimum intensity value
				float min_intensity_value = 0;
				// Actual (first) level maximum intensity value
				float max_intensity_value = intensityIncrement;
				// Actual (first = 0) level
				int currentLevel = 0;

				currentIndex[0] = row;
				currentIndex[1] = col;

				OutputImageType::PixelType currentValue = rescaleFilter->
					GetOutput()->GetPixel(currentIndex);

				while (pixel_classified == false && min_intensity_value <=255)
				{

					// If the scanned pixel intensity value is in the actual
					// intensity level range
					if (currentValue >= min_intensity_value && 
					    currentValue < max_intensity_value)
					{

						// Increment the pixel count for this level by one
						intensityCount[currentLevel]++; 
						// and set the flag to pixel being classified
						// (this ends the while loop and starts a new 
						// row/column scan in the for cycle
						pixel_classified = true;

					}

					else
					{

						// If not, I pass to the next intensity level,
						// setting the next minimum level range and maximum level
						// range
						min_intensity_value += intensityIncrement;
						max_intensity_value += intensityIncrement;
						currentLevel++;

					}
				}


			}	   // for columns close
		}	   // for rows close



		// An output to control the histogram values, commented because very long
		/*
		std::cout << std::endl << "          Intensity count for each intensity level:  " << std::endl << std::endl;
		for (int jj = 0; jj <= levelsNumber-1; jj++)
		{
			std::cout << "           [l" << jj << "] " << intensityCount [jj] << std::endl;
		}
		*/

		int maxIndex = std::distance(intensityCount.begin(), std::max_element(intensityCount.begin(), intensityCount.end()));
		// std::cout << std::endl << "          Level index in wich the intensity count is maximum: " << maxIndex << std::endl;

		// Here I calculate the dishomogeneity index for each level of each 
		// crosscorrelation image.
		// I use an inversed gaussian function (or a x-specular sigmoid function)
		// centered in the level index in wich the intensity count is maximum,
		// multiplied for the pixel count of each level.
		// The aim is to weight less the pixel count near the center (level
		// index in which the intensity count is maximum) where the pixels are
		// considered homogeneous, and to weight more the pixel count in the levels
		// far from the center, where the pixels are considerer to give dishomegeneity
		// to the crosscorrelation image.

		// Here, "x" is exponent of the exponential (sigmoid) function
		float x = -6;
		// Here I store the corection factor based on dishomogeneity
		std::vector<float> correctionFactor (levelsNumber , 0);

		// Here I set a for cycle starting for the "central" intensity level,
		// to the first ( = 0) intensity level
		for (int jj = maxIndex; jj >= 0; --jj)
		{
			// This is the sigmoidal function for the left part of the histogram
			// The first term defines the amplitude
			correctionFactor[jj] =  100 / ( 1 + exp (-x) );
			// Then, I set the next step of the sigmoid function, for the next
			// intnsity level (exponential becomes more positive, so the
			// function value is higher)
			x += 0.25;
		}
		x = -6;

		// Here, it's the same, but for the right part of the histogram, from the
		// "center" to the last intensity level (bins + 1)
		for (int jj = maxIndex + 1; jj <= levelsNumber-1; jj++)
		{
			correctionFactor[jj] =  100 / ( 1 + exp (-x) );
			x += 0.25;
		}

		/* ! Be carfull, exponent, amplitude, and exponent increment very affect
		 the dishomogeneity evaluation of the crosscorrelation images. They
		 must be choosed carefully ! */

		// Here I calculate the the dishomogeneity value for each level,
		// multipling the intensity count for the correctin factor, for
		// each level
		std::vector<float>  correctedCount (levelsNumber , 0);
		for (int jj = 0; jj<=levelsNumber-1; jj++)
		{
			correctedCount [jj] = intensityCount [jj] * correctionFactor [jj];

			/*
			std::cout << "          correction factor is:    " <<correctionFactor [jj] << std::endl;
			std::cout << "          new corrected vector is:    " <<correctedCount [jj] << std::endl;
			*/
		}


		// Here I calculate the overall dishomogeneity factor for each
		// crosscorrelation image
		float correctedCount_sum = std::accumulate(correctedCount.begin(),correctedCount.end(),0);

		homogeneityValues.push_back(correctedCount_sum);

		/*
		for (int jj = 0; jj <= homogeneityValues.size() - 1; jj++ )
		{
		std::cout << "         Dishomogeneity vector is    " << homogeneityValues[jj] << std::endl;
		}
		*/



		/*======================================================================
		  
		 Clock Probe
		  
		 =====================================================================*/
		// Here I estimate the time needed to complete process all the slices

		int time;				// Defining the time variable 
		int min;				// Defining the minutes variable 
		int sec;				// Defining the seconds variable 

		// Calcoating the remaining time to process all slices
		// Stopping the clock probe function to get the time in this moment
		clock.Stop();

		// I calculate the mean (of all already completed cycle) at every cycle,
		// to get a more precise estimation.
		// "clock.GetMean" gives the mean time of all cycles
		time = ( clock.GetMean() * ( (size_z - k) / sample_interval) );
		if ( time > 60)
		{
			min = time / 60;
			sec = time - (min*60);
			
			if (output_mode == 0)
			{
				std::cout << std::endl << "     ~ Estimated time to process all remaining slices: " 
					<< min << " min " << sec << " sec" << std::endl << std::endl;
			}
			else
			{
				std::cout << "\r" << "  |" << std::flush;
				for (int  c = 0; c <= k; c++)
				{
					std::cout << "==>|" << std::flush;

				}
				for (int  c = 0; c < size_z - k; c++)
				{
					std::cout << "   |" << std::flush;

				}
				std::cout << "  (" 
					<< min << " min " << sec << " sec left)           " << std::flush;
			}
		}
		else
		{
			sec = time;

			if (output_mode == 0)
			{
				std::cout << std::endl << "     ~ Estimated time to process all remaining slices: " 
					<< sec << " sec"<< std::endl << std::endl;
			}
			else
			{
				std::cout << "\r" << "  |" << std::flush;
				for (int  c = 0; c <= k; c++)
				{
					std::cout << "==>|" << std::flush;

				}
				for (int  c = 0; c < size_z - k; c++)
				{
					std::cout << "   |" << std::flush;

				}
				std::cout << "  (" 
					<< sec << " sec left)           " << std::flush;
			}
		}


	}   // Crosscorrelation for cycle close
	//---------------------- CROSSCORRELATION FOR END ----------------------//

	std::cout << "          Done! " << std::endl;
	



    /*=========================================================================

    Non-prostatic slices exclusion

    =========================================================================*/
	// Here I set up a method to exclude the slices without prostate.
	// I use the standard deviation intensity, mean intensity, dishomogeneity
	// intensity information calculated before on the crosscorrelation images.

	// By default, I include all the slices in the volume
	int upper_slice_include_value = size_z;
	int lower_slice_include_value = 1;

	// Flags for th success/insuccess of the method
	bool upper_limit_found = false;
	bool lower_limit_found = false;


	// I need to calculate the normalized vector fo the followings:
	std::vector<float>  norm_homogeneityValues (size_z , 0);
	std::vector<float>  norm_sd (size_z , 0);
	std::vector<float>  norm_meani (size_z , 0);

	// The normalized values must be from 0 to 1

	// Normalizing to the minimum
	float min_h = *std::min_element(homogeneityValues.begin(), homogeneityValues.end());
	float min_sd = *std::min_element(sd.begin(), sd.end());
	float min_meani = *std::min_element(meani.begin(), meani.end());

	for (int j = 0; j <= size_z - 1; j++)
	{
		norm_homogeneityValues [j] = ( homogeneityValues[j] / min_h ) - 1;
		norm_sd [j] = ( sd[j] / min_sd ) - 1;
		norm_meani [j] = ( meani[j] / min_meani ) - 1;
	}

	// Normalizing to the maximum
	float max_h = *std::max_element(norm_homogeneityValues.begin(), norm_homogeneityValues.end());
	float max_sd = *std::max_element(norm_sd.begin(), norm_sd.end());
	float max_meani = *std::max_element(norm_meani.begin(), norm_meani.end());

	for (int j = 0; j <= size_z - 1; j++)
	{
		norm_homogeneityValues [j] = ( norm_homogeneityValues[j] / max_h );
		norm_sd [j] = ( norm_sd[j] / max_sd );
		norm_meani [j] = ( norm_meani[j] / max_meani );
	}

	if (output_mode == 0)
	{
		std::cout << std::endl << "         Normalized Dishomogeneity values are: " << std::endl << std::endl;
		for (int j = 0; j <= size_z - 1; j++)
		{
			std::cout <<  "         nDish ["<< j + 1 << "] " << norm_homogeneityValues[j] << std::endl;
		}

		std::cout << std::endl << "         Normalized Standard Deviation values are: " << std::endl << std::endl;
		for (int j = 0; j <= size_z - 1; j++)
		{
			std::cout << "         nSD ["<< j + 1 << "] " << norm_sd[j] << std::endl;
		}

		std::cout << std::endl << "         Normalized Mean values are: " << std::endl << std::endl;
		for (int j = 0; j <= size_z - 1; j++)
		{
			std::cout << "         nMean ["<< j + 1 << "] " << norm_meani[j] << std::endl;
		}
	}

	// Here I calculate the sum of all of the three, to get an overall
	// estomation of the crosscorrelation image.
	// Higher values stand for a more sparse intensity values of the
	// crosscorrelation image, i.e. a less correlation of the actual slice
	// with the central one, wich has the best prostatic profile.
	// So I assume a less probability to find the prostate in the actual slice.

	std::vector<float>  norm_sum (size_z , 0);
	if (output_mode == 0)
	{
		std::cout << std::endl << "         Estimates Uncorrelation Factors are: " << std::endl;
	}

	for (int j = 0; j <= size_z - 1; j++)
	{
		norm_sum[j] = (norm_homogeneityValues [j] + norm_sd [j] + norm_meani [j]) / 3;
		if (output_mode == 0)
		{
			std::cout << "         ["<< j + 1 << "] " << norm_sum[j] << std::endl;
		}

	}
	
	// Here I proceed to set a relative threshold to cut of the trend of the
	// uncorrelation function previously calculated.
	// The treshold is relative to the maximum value for each on the two
	// limb of the function (becouse right and left part of the funcion
	// have very different trends)

	// Maximum in the left part
	float max_lower = norm_sum[0];
	float max_lower_index;

	for (int j=0; j< central_slice-1; j++)
	{
		if (max_lower < norm_sum[j])
		{
			max_lower = norm_sum[j];
			max_lower_index = j;
		}
	}

	// Maximum in the righ part
	float max_upper = norm_sum[central_slice + 1];
	float max_upper_index;

	for (int j=central_slice; j<=size_z - 1; j++)
	{
		if (max_upper < norm_sum[j])
		{
			max_upper = norm_sum[j];
			max_upper_index = central_slice - (j - (central_slice - 1));
		}
	}

	// Since the maximum value may be not at the first/last slice (that is the
	// beahvioure I expect) I must re-calculate the lower and upper
	// threshold in function of their distance from the first and last
	// slice respectively
	// ex.: a function monotonly crescent till a certain point, and then
	// monotonly decrescent. The threshold based on this maximum will be
	// wrong and understimate. I have to correct, i.e. increase, the maximum value.

	float new_max_lower =  ( ( max_lower_index / central_slice ) * max_lower ) + max_lower;
	float new_max_upper =  ( ( max_upper_index / central_slice ) * max_upper ) + max_upper;

	// Here the treshold is set to a fixed percentage of the new calculated 
	// maximum for each limb of the uncorrelation function

	float lower_th = 0.75 * new_max_lower;	  // default = 75%
	float upper_th = 0.75 * new_max_upper;	  // default = 75%

	if (output_mode == 0)
	{
		std::cout << std::endl << "         Bottom threshold is: " << lower_th << std::endl;
		std::cout << "         Top threshold is: " << upper_th << std::endl;
	}


	for (int j = central_slice - 1; j >= 0; --j)
	{
		if (norm_sum[j] > lower_th)
		{
			lower_slice_include_value = (j + 1) + 1;
			lower_limit_found = true;
			break;
		}
	}

	for (int j = central_slice - 1; j <= size_z - 1; j++)
	{
		if (norm_sum[j] > upper_th)
		{
			upper_slice_include_value = (j + 1) - 1;
			upper_limit_found = true;
			break;
		}
	}

	std::cout << std::endl << "        · Bottom first (z axis) included slice value is: " << lower_slice_include_value << std::endl;
	std::cout << "        · Top last (z axis) included slice value is: " << upper_slice_include_value << std::endl;



	/*=========================================================================
	  
	 dcm2nii_dwi - Cropped (pre-processed) b0 volume
	  
	 =========================================================================*/

	// Now I proceed to rebuild the volume from the slices previously generated.
	// Since the dcm writing cause a lost of some dicom information, 
	// I use the slices' file names generated to rebuild the volume. 

	d2n_reader->SetImageIO( dicomIO );

	nameGenerator->SetUseSeriesDetails( false );
	nameGenerator->SetDirectory( cropped_dir );
	std::string prostate_volume;
	try
	{
		if (output_mode == 0)
		{
			std::cout << std::endl << "          Analizing directory..." << std::endl;
			std::cout << "          Found following DICOM Series: ";
			std::cout << std::endl;
		}

		typedef std::vector< std::string >    SeriesIdContainer;
		typedef std::vector< std::string >   FileNamesContainer;
		FileNamesContainer fileNames;

		const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

		SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
		SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
		while( seriesItr != seriesEnd )
		{
			if (output_mode == 0)
			{
				std::cout << "          " << seriesItr->c_str() << std::endl;
			}
			fileNames.insert(fileNames.end(), nameGenerator->GetFileNames(seriesItr->c_str()).begin(), nameGenerator->GetFileNames(seriesItr->c_str()).end());
			seriesItr++;
		}

		std::sort(fileNames.begin(), fileNames.end(), StringLessThen);

		d2n_reader->SetFileNames( fileNames );

		try
		{
			d2n_reader->Update();
		}
		catch (itk::ExceptionObject &ex)
		{
			std::cout << ex << std::endl;
			return EXIT_FAILURE;
		}

		typedef itk::OrientImageFilter<ImageType,ImageType> OrienterType;
		OrienterType::Pointer orienter = OrienterType::New();

		orienter->UseImageDirectionOn();
		orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
		orienter->SetInput(d2n_reader->GetOutput());

		typedef itk::ImageFileWriter< ImageType > WriterType;
		WriterType::Pointer writer = WriterType::New();

		// Generating volume output file name, usign the output directory given

		std::stringstream out_dir;
		out_dir << cropped_dir;				// setting the given output directory
		prostate_volume = out_dir.str();		// passing output directory to "tmp_dir_volume"
		prostate_volume += "/Volume_Prostate.nii";		// post-adding "/---.nii"

		writer->SetFileName( prostate_volume );

		writer->SetInput( orienter->GetOutput() );

		if (output_mode == 0)
		{
			std::cout << std::endl << "          Generating the new volume ... " << std::endl;
			std::cout << "          New volume saved as: " << std::endl;
			std::cout << "          " << prostate_volume << std::endl;
			std::cout << "          Done!" << std::endl;
		}

		try
		{
			writer->Update();
		}
		catch (itk::ExceptionObject &ex)
		{
			std::cout << ex << std::endl;
			return EXIT_FAILURE;
		}
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}

	

	// --- -- -							↓							 - -- --- //
	// The following part is commented becouse with the latest version of the code
	// is considered to generate too much instable results.
	// Instead another approach is applied and described in the following
	// H automatic search part of the code.
	// However, this part is nor removed, ready to be reverted and used if needed
	// (the part of the code the estimates radius, and (x,y) center coordinates
	// of the prostate in the H search algorithm must be commented)

	/*=========================================================================

    Multiple Initialization parameters extraction with Hough transform

    =========================================================================*/
	// Here I get some paremeters to be used to achieve multiple initialization
	// of the levelset.
	// I use Hough trasform to find a circle on the cropped images (containing
	// only the prostate) to finda an aproximative profile of the prostate,
	// and I use the radius and the center coordinates of this
	// circle to extraploate the parameters I need.
	// I use only the central slice.

	// I instantiate some variables
	float x_center;
	float y_center;
	float radius;


	//  Next, I declare the pixel type and image dimension and specify the
	//  image type to be used as input. I also specify the image type of the
	//  accumulator used in the Hough transform filter.

	typedef   float   HoughPixelType;
	typedef   float           AccumulatorPixelType;

	typedef itk::Image< HoughPixelType, 2 >  HoughImageType;
	HoughImageType::IndexType localIndex;
	typedef itk::Image< AccumulatorPixelType, 2 > AccumulatorImageType;  

	//  I create the HoughTransform2DCirclesImageFilter based on the pixel
	//  type of the input image (the resulting image from the
	//  ThresholdImageFilter).
	typedef itk::HoughTransform2DCirclesImageFilter<HoughPixelType,
	AccumulatorPixelType> HoughTransformFilterType;
	HoughTransformFilterType::Pointer houghFilter = HoughTransformFilterType::New();
	
/*	 
 	std::cout << std::endl << "→ Now Multiple Initialization parameters extraction is going to be performed ... " << std::endl;
	 
	// Generating the file names to be read, so I can keep the cropped slices
	// from the cropped folder.
	// I set the slice number to the central one, becouse I use only the
	// central slice to extrapolate the values I need.
	std::string Hough_sn;
	std::stringstream Hough_out_slice;
	Hough_out_slice << central_slice;
	Hough_sn = Hough_out_slice.str();

	std::string b0_dir_slice;
	b0_dir_slice += cropped_dir;
	b0_dir_slice += "/slice_cropped";
	// I want a name generation like this:
	// slice01, slice02, ... , slice10, slice11, ...
	if ( central_slice < 10 )
	{
		b0_dir_slice += "0";
	}
	b0_dir_slice += Hough_sn;
	b0_dir_slice += ".dcm";	

	std::cout << std::endl << "     → Now processing slice " << central_slice << std::endl;


	// Instantiatine the reader
	typedef itk::ImageFileReader< OutputImageType > Hough_ReaderType;
	Hough_ReaderType::Pointer Hough_reader = Hough_ReaderType::New();

	std::cout << std::endl << "          → Loading candidate key slice: " << b0_dir_slice << std::endl;
	
	Hough_reader -> SetFileName(b0_dir_slice);
	Hough_reader -> Update();
	
	std::cout << "            Done!" << std::endl;
	std::cout << "          → Performing Hough transform ... " << std::endl;

	//  I set the input of the filter to be the output of the
	//  ImageFileReader. I set also the number of circles I are looking for.
	//  Basically, the filter computes the Hough map, blurs it using a certain
	//  variance and finds maxima in the Hough map. After a maximum is found,
	//  the local neighborhood, a circle, is removed from the Hough map.
	//  SetDiscRadiusRatio() defines the radius of this disc proportional to
	//  the radius of the disc found.  The Hough map is computed by looking at
	//  the points above a certain threshold in the input image. Then, for each
	//  point, a Gaussian derivative function is computed to find the direction
	//  of the normal at that point. The standard deviation of the derivative
	//  function can be adjusted by SetSigmaGradient(). The accumulator is
	//  filled by drawing a line along the normal and the length of this line
	//  is defined by the minimum radius (SetMinimumRadius()) and the maximum
	//  radius (SetMaximumRadius()).  Moreover, a sweep angle can be defined by
	//  SetSweepAngle() (default 0.0) to increase the accuracy of detection.


	double NumberOfCircles;
	HoughImageType::Pointer localImage;

	// Here I define the number of circle to be find,
	// and I link the pointer to the correct image for each case. 

	NumberOfCircles = 1;		// adding the new folder
	localImage =     Hough_reader -> GetOutput();


	// Here I set the others non variable hough parameters
	double MinimumRadius = 30;
	double MaximumRadius = 60;
	double SweepAngle = 0.6; 		// default = 1
	//double SigmaGradient = 100;		// default = 1
	//double Variance = 5;		// deafult = 5;
	//double DiscRadiusRatio = 10;	// deafult = 10;

	houghFilter->SetInput( localImage );

	houghFilter->SetNumberOfCircles( NumberOfCircles );
	houghFilter->SetMinimumRadius(   MinimumRadius );
	houghFilter->SetMaximumRadius(   MaximumRadius );
	houghFilter->SetSweepAngle( SweepAngle );
	//houghFilter->SetSigmaGradient( SigmaGradient );
	//houghFilter->SetVariance( Variance );
	//houghFilter->SetDiscRadiusRatio( DiscRadiusRatio );

	houghFilter->Update();
	OutputImageType::Pointer localAccumulator = houghFilter->GetOutput();   

	//  I can also get the circles as \doxygen{EllipseSpatialObject}. The
	//  \code{GetCircles()} function return a list of those.

	HoughTransformFilterType::CirclesListType circles;
	circles = houghFilter->GetCircles( NumberOfCircles );

	//  I can then allocate an image to draw the resulting circles as binary
	//  objects.

	typedef    short   HoughOutputPixelType;			// unsigned char //
	typedef  itk::Image< HoughOutputPixelType, 2 > HoughOutputImageType;  

	HoughOutputImageType::Pointer  localOutputImage = HoughOutputImageType::New();

	HoughOutputImageType::RegionType region;
	region.SetSize(localImage->GetLargestPossibleRegion().GetSize());
	region.SetIndex(localImage->GetLargestPossibleRegion().GetIndex());
	localOutputImage->SetRegions( region );
	localOutputImage->SetOrigin(localImage->GetOrigin());
	localOutputImage->SetSpacing(localImage->GetSpacing());
	localOutputImage->Allocate();
	localOutputImage->FillBuffer(0);


	//  I iterate through the list of circles and I draw them.

	typedef HoughTransformFilterType::CirclesListType CirclesListType;
	CirclesListType::const_iterator itCircles = circles.begin();

	// Here I get the (x,y) center coordinates and radius
	x_center = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
	y_center =  (*itCircles)->GetObjectToParentTransform()->GetOffset()[1]; 
	radius = (*itCircles)->GetRadius()[0];

	std::cout << "            Circle (x) coordinate is: " << x_center << std::endl;
	std::cout << "            Circle (y) coordinate is: " << y_center << std::endl;
	std::cout << "            Circle radius is:         " << radius << std::endl;
	std::cout << "            Done!" << std::endl << std::endl;

	x_center = size[0] / 2;
	y_center =  (size[1] / 2 ) + (size[1] / 2 )*0.08;

	std::cout << "            Default 'x' coordinate is: " << x_center << std::endl;
	std::cout << "            Default 'y' coordinate is: " << y_center << std::endl;
	*/

	// Now I extrapolate the new seeds to be passed to the levelset
	// float in needed becouse int cause too mauch error in further calculations.
	float x_left[2];
	float x_right[2];
	float y_top[2];
	float y_diag_left[2];
	float x_diag_left[2];
	float y_diag_right[2];
	float x_diag_right[2];
	float y_diag_left_bis[1];
	float x_diag_left_bis[1];
	float y_diag_right_bis[1];
	float x_diag_right_bis[1];



    /*=========================================================================

    Optimal Levelset Heaviside function (H, "epsilon") value automatic search 

    =========================================================================*/
	// Here I try to predict the optimal heaviside function for the Chan and Vese
	// levelset.
	// The heaviside value, "epsilon", very influence the levelset behaviour and
	// segmentetion result.
	// A lower value of H implies a overstimated segmentation, on the opposite,
	// an higher value of H implies an understimated segmentation.
	// So I need a way to estimate the levelset performances, thus implies a
	// method to evaluate the prostate surface extension on the MRI, and compare
	// it to a parameter that estimate the levelset surface.
	//
	// Here I implement the following method:
	// the estimtion of the prostate extension (surface or volume) is actuated
	// identifing the foreground pixels and the background pixels on both MR 
	// images and segmentede images. Foreground pixels count of foreground on MRI
	// must match with foreground pixels count on segmented images.
	// Only foreground pixels are taken in account, becouse I must be sure
	// that the segmented pixels count is not less than the MRI count ( I want
	// to be sure to not uderstimate the segmentation). This is the reason way
	// a parementer with mixed count of back and foreground is not implemented.
	// The search moves throught the mimimization of the error given by the
	// difference of the pixel cont between foreground of the MR and segmented
	// images.
	// Ideally, this error is 0. In practice I try to predict the value of H 
	// that gives a error of 0.
	// The back and foreground distinguish is actuated by classification of
	// pixels intensity, counting the number of pixels up or down a fixed 
	// threshold, ideally dividing back and foreground values.
	// Basically is a historam abnd division. To obtain this, a modified version
	// of the previously used dishomogeneity code is used.
	// To obivious reasons, this distinguish is performed on b0 images, and the
	// original (optimal) count is obtained.
	// The H serach is performed in the following way:
	// a start H value is manually fixed. Then the Chan and Vese levelset 
	// is performed. Than the segmentation image is loaded and evalueated as 
	// described ( comparing original foreground count to segmented count, and
	// calculatin the relative error). After that, a second segmentation is 
	// performed, with an increased (different) H value, by a big step, so I am
	// performin a first rough search. Now, I know two error values and their
	// relative H values. With them I perform a linear interpolation to 
	// predict the value of H in which the error is 0 (linear prediction).
	// The linear predicted H value is used to perform a third segmentation, and
	// its real error is estimated, than a forth segmentation is performed,
	// with an H value increased by a little step, to perform a fine search 
	// (becouse now H is very close to the 0 error value, and due to is non linear
	// behaviour, a search in is neighbour is more preise). So the last linear
	// prediction of the H value thath gives a 0 error is performed. This is the
	// final Heaviside (epsilon) function value, and a last (fifth) segmentation
	// is performed.
	// (this is my implementaion of the my idea, a more precise implementation
	// of the search algorithm may be written, but this implementation is a
	// good compromise of performance / precision)
	// Note: an important paramenter to take in account is in wich slices or
	// portion of the volume the classification (histogram) is performed.
	//
	// To achieve a more precise segmentaton, the whole prostate volume is diveded
	// in three sub-volumes: the top part (the one with the bladder), the central
	// one, and the bottom one. For each sub-volume I perform the H search and
	// the sub-final Chan and vese levelset segmentation. So, at the endo of the
	// search algorithm, I obtain 3 segmented volume. Each of this mask is
	// the complete 3D segmentation, i.e. containt the whole volume, but only
	// the part in wich the H search has been performed (i.e. the sub-voluyme taken
	// in account) rapresent the optimal segmentation with the optimal H value.
	// The reason why I obtain 3 complete volumes, instead 3 sub-volumes, is
	// because I alwayse intitalize on the central slice, then the segmentation
	// propagates throught the whole volume. So, only the part of the volume
	// I'm scanning for H is the optimal segmentation.
	// This prevent to deal with the difficult problem to initialize the levelset
	// in others part of the volume.
	// After that, the reconstruction algorithm will rebuil the whole correct 
	// segmented volume taking each slice from the correct segmented volume (top,
	// central, bottom).

	std::cout << std::endl << "→ Now Automatic Heaviside Function Search Algorithm is going to be performed ... [9/19]" << std::endl;

	// The classification is performed in 3D
	const unsigned int CnV_Dimension = 3;	
	typedef float ScalarPixelType;
	typedef itk::Image< ScalarPixelType, CnV_Dimension > InternalImageType;

	typedef itk::ImageFileReader< InternalImageType > CnV_ReaderType;
	CnV_ReaderType::Pointer CnV_reader = CnV_ReaderType::New();

	// The path of the segmented volume saved by the writer at the end of the
	// Chan and Vese levelset, that must be loaded to be evaluated adn that
	// must be saved to successive rebuilt, from 3 volumes to one.

	std::string segmented_vol;
	std::string central_segmented_vol;
	std::string top_segmented_vol;
	std::string bottom_segmented_vol;

	// Here I load the previously save maximum intensity for each b0 slice
	// into a temporary variable, that idicates the maximum intensity for
	// each sub-volume.
	// The maximum intensity taken in consideration is the one in the central slice
	// of the sub-volume, for each sub-volume.
	// For an example: if I'm scanning the sub-volume from slice 10 to 14, the 
	// maximum intensity is take into slice 12.
	// This imply that the maximum precision is reached in the central slice
	// of each sub-volume.
	// NOTE: using the maximum intensity value of the whole volume will 
	// generate a very high error in the prediction algorithm, since
	// each slice has very different intensity range. For this reason,
	// the height (slices interval) of the sub-volumes must be choosen
	// very carefully.
	
	float maximum_intensity = maximum_intensity_b0 [central_slice-1];

	// Current pixel index I am scanning
	InternalImageType::IndexType currentIndex;
	// Histogram bins (in wich I divide the intensity range)
	float histogramBins = maximum_intensity / 2;
	// Number of effective intensity levels ("+1" to take in account of
	// the pixel values out of range of the last bin)
	int levelsNumber = histogramBins + 1;
	// The pixel range of each bins
	float intensityIncrement = maximum_intensity / histogramBins;
	// A vector to store the cont of each pixel intensity, in every level
	std::vector<float> intensityCount (levelsNumber , 0);

	// Instantiating teh MRI background value and foreground value
	float background_original_count = 0;
	float foreground_original_count = 0;

	// Instantiating the vectors containing the count of MRI background 
	// and foreground pixels intensity for each iteration
	std::vector<float> background_masked_count (5 , 0);
	std::vector<float> foreground_masked_count (5 , 0);

	// The same for the error and epsilon (H)
	std::vector<float> foreground_error (5 , 0);
	std::vector<float> epsilon (5 , 0);

	// An origin used in the change information filter that is reused and must be 
	// defined outside a steatment
	InternalImageType::PointType iorigin;
	// Curret pixel intensity value at the curretn index
	InternalImageType::PixelType currentValue;
	// Pixel intensity value of the segmented image (0-1)
	// If 0, the bacground counter is incremented, else the foreground
	// counter is incremented
	InternalImageType::PixelType switchValue;

	// The Linear Interpolation prediction of the epsilon (H) value where
	// the estimated error is 0
	float epsilonzeroLI;

	// A versor to determinate the direction of the increment in the H steps search
	// (+1: increment, -1: decrement).
	// this is necessary to drive the H search to the 0 error direction,
	// before the linear prediction is actuated.
	int dir_epsilon;

	// The search start value. This is fixed manually, after a brief manually 
	// search of the optimal-like H value for a wide range of patient of the same
	// device. Is extremly important to start from a value nearest as possible
	// to the optimal one found by the search algorithm.
	// Example: start = 500. H range from the search is above: 300 - 1200.
	// This increment the precision of the algorithm and lowers the possibility
	// of a search fail.
	// This value will be also used in case of failure in the search process,
	// as a security backup value.
	float startH_value = 500;

	// This variable is used to indicate wich sub-volume I am scanning
	// 0 (start): central
	// 1		: bottom
	// -1		: top
	int vol_part = 0;

	// A flag for the search fail
	bool H_search_fail;
	bool H_search_fail_central;
	bool H_search_fail_top;
	bool H_search_fail_bottom;

	// This flag activates the radius calculation in the first cycle, and prevents
	// it in the others cycles.
	// The radius estimation is no loger rely on the hough transform (as a previously
	// implementation of the code) because the estimation is too much instable,
	// and combined with this H search (wich has a good accuration) oftem drive
	// the segmentation to errors, expecially understimation of the segmentation
	// (due to too far positionament of the seed from the boundary of the coil).
	bool Do_radius = true;

	
	// Reader of the saved segmented volume
	CnV_ReaderType::Pointer H_reader = CnV_ReaderType::New();

	if (output_mode == 0)
	{
		std::cout << std::endl << "          → Loading prostatic volume to be segmented: " << prostate_volume << std::endl;
	}
    CnV_reader -> SetFileName(prostate_volume);	
    CnV_reader -> Update();

	// --- -- - H SEARCH FOR START - -- --- //
	// As decribed before, the automatic search runs for 6 iterations
	for (int iter = 1; iter <= 6; iter++)

	{
		itk::TimeProbe H_clock;
		H_clock.Start();
		
		// --- -- - VOLUME SCAN FOR START - -- --- //
		// Here I scan every row and every column and I classify each pixel for
		// each cycle (row and col remains fixed until the pixel is classified)
		// In z direction, I scan only near the central slice of the sub-volume
		// I am currently scanning,because the external slice generates a not
		// properly correct H search, because here the levelset is too much
		// imprecise.

		// Since vol_part changes from 0 → 1 → -1, it drives the z scan amplitude

		for (int height = ((central_slice - 1) - 2 ) + ( vol_part * 5 );
		     height <= ( (central_slice - 1) + 2 ) + ( vol_part * 5 );
		     height++)
		{			
			for (int row = 0; row <= size[0]; row++)
			{
				for (int col = 0; col <= size[1]; col++)
				{

					// Every new cycle I must redefine:
					// Flag set to pixel being not classified
					bool pixel_classified = false;
					// Actual (first) level minimum intensity value
					float min_intensity_value = 0;
					// Actual (first) level maximum intensity value
					float max_intensity_value = intensityIncrement;
					// Actual (first = 0) level
					int currentLevel = 0;

					currentIndex[0] = row;
					currentIndex[1] = col;
					currentIndex[2] = height;

					if (iter == 1)
					{
						// If is the first iteration, the b0 MRI is scanned, to 
						// search the optimal (original) back and foreground
						// pixels count

						currentValue = CnV_reader->GetOutput()->GetPixel(currentIndex);

					}
					
					if (iter > 1)
					{

						// If is not the first iteration, the segmented volume
						// is scanned, to evaluate back and foreground pixels 
						// count
						
						H_reader->SetFileName(segmented_vol);
						H_reader->Update();

						switchValue = H_reader->GetOutput()->GetPixel(currentIndex);

					}
					

					// If is the first iteration, the b0 MRI is scanned, to 
					// search the optimal (original) back and foreground
					// pixels count
					if (iter == 1)
					{	

						while (pixel_classified == false && min_intensity_value <=maximum_intensity)
						{

							// If the scanned pixel intensity value is in the actual
							// intensity level range

							if (currentValue >= min_intensity_value && currentValue < max_intensity_value)
							{

								// Increment the pixel count for this level by one
								intensityCount[currentLevel]++; 
								
								// and set the flag to pixel being classified
								// (this ends the while loop and starts a new 
								// row/column scan in the for cycle
								pixel_classified = true;

							}

							else
							{

								// If not, I pass to the next intensity level,
								// setting the next minimum level range and maximum level
								// range
								min_intensity_value += intensityIncrement;
								max_intensity_value += intensityIncrement;
								currentLevel++;

							}
						}	   // while close

					}	   // if (iter = 1) close

					// If I am not in the first iteration
					else
					{
						if (switchValue == 0)
						{
							// Out of mask case, the pixel is classified as 
							// background
							background_masked_count[iter-2] += 1;
						}
						else
						{
							// Inside mask case, the pixel is classified as 
							// foreground
							foreground_masked_count[iter-2] += 1;
						}

					}	   // else (iter = 1) close

				}	   // for columns close
			}	   // for rows close
		}	   // for height close
		
		// If is the first iteration, the b0 MRI is scanned, to 
		// search the optimal (original) back and foreground
		// pixels count. Here the classification is performed
		if (iter == 1)
		{

			for (int jj = 0; jj <= levelsNumber-1; jj++)
			{
				// Classifcation
				// The threshold divides the levels into background and foreground
				// The threshold is relative to the maximum intensity (because
				// levelsNumber is relative to the maximum intensity), so
				// this value may be a good global value, but is important
				// to be sure that fits the features of every MRI device.
				if ( jj <= ( (levelsNumber / 10 ) - 1 ) )  // This is the threshold
				{
					background_original_count += intensityCount [jj];
				}
				else
				{
					foreground_original_count += intensityCount [jj];
				}

			}

			if (output_mode == 0)
			{
				std::cout << std::endl << "          → Analyzing ... ";
				std::cout << std::endl << "            Optimal background pixel count estimated is: " << background_original_count ;
				std::cout << std::endl << "            Optimal foreground pixel count estimated is: " << foreground_original_count << std::endl;
			}

			// In the first cycle, teh radius estimation is performed.
			// Here I present a stable way (more than hough) but less precise.
			// Method: The foreground pixel count of the sub-volume is divided 
			// by the slice scanning height, to obtainthe mean foreground of
			// one slice. This rapresent the mean surface of the prostate in this
			// sub-volume.
			// The prostate is assumed as a perfect circle, so the radius can be
			// obtained.
			// than a reduction in radius is performed, to take in account of the
			// foreground classified pixels that are not part of the prostate.
			if (Do_radius == true)
			{

				if (output_mode == 0)
				{
					std::cout << std::endl << "→ Now Multiple Initialization parameters extraction is going to be performed ...  [10/19]" << std::endl;
				}

				// Also the prostate center coordinates are no more extrapolated
				// using hough, due to high instability.
				// x is assumed as the center of the image
				// y is al littlke more under the center of the image.
				// Of course, in this case teh prostate must be well centered
				// on examination by the operator.
				// Anyhow, the hough method can be reverted, and is still present
				// in this code.
				x_center = size[0] / 2;
				y_center =  (size[1] / 2 ) + (size[1] / 2 )*0.08;

				if (output_mode == 0)
				{
					std::cout << "          Default 'x' coordinate is: " << x_center << std::endl;
					std::cout << "          Default 'y' coordinate is: " << y_center << std::endl;
				}
				
				// Surface extraction
				float prostate_surface = foreground_original_count / 5;
				// Radius calculation
				radius = sqrt(prostate_surface / 3.14);
				// Reduction of 15%
				radius = radius - (radius/100*15);
				if (output_mode == 0)
				{
					std::cout << std::endl << "          Estimated radius is: " << radius << std::endl;

					// Now the seeds to the multiple initialization can be calculated
					std::cout << "          → Calculating new seeds ... " << std::endl;
				}


				// Here I represent the way I take the points.
				// I follow a radial way.
				// Below are represented the directions from the center of the circle.
				// Fore every segment I take two points (a couple of coordinates).
				// The segments maximum lenght is the circle radius.
				// I take point at 4/10 and 8/10 of the radius.
				// 8/10 becouse I don't want to risk to fall out of the prostate.
				// 4/10 becouse the center is'n so much significative to be initialized.
				//		\  |  /
				//		 \ | /
				//	 _____\|/_____

				// Left horizontal segment
				x_left[0] = x_center - (radius / 10 * 8);
				x_left[1] = x_center - (radius / 10 * 4);

				// Right horizontal segment
				x_right[0] = x_center   + (radius / 10 * 4);
				x_right[1] = x_center   + (radius / 10 * 8);

				// Top central verical segment
				y_top[0] = y_center - (radius / 10 * 4);
				y_top[1] = y_center - (radius / 10 * 8);

				// Left diagonal 45° segment
				y_diag_left[0] = y_center - (radius * (sqrt(2)/2) / 10 * 4);
				x_diag_left[0] = x_center - (radius * (sqrt(2)/2) / 10 * 4);
				y_diag_left[1] = y_center - (radius * (sqrt(2)/2) / 10 * 9);
				x_diag_left[1] = x_center - (radius * (sqrt(2)/2) / 10 * 9);

				// Left diagonal 30° segment
				y_diag_left_bis[0] = y_center - (radius * (1/2) / 10 * 9);
				x_diag_left_bis[0] = x_center - (radius * (sqrt(3)/2) / 10 * 9);

				// Right diagonal 45° segment
				y_diag_right[0] = y_center - (radius * (sqrt(2)/2) / 10 * 4);
				x_diag_right[0] = x_center + (radius * (sqrt(2)/2) / 10 * 4);
				y_diag_right[1] = y_center - (radius * (sqrt(2)/2) / 10 * 9);
				x_diag_right[1] = x_center + (radius * (sqrt(2)/2) / 10 * 9);

				// Right diagonal 30° segment
				y_diag_right_bis[0] = y_center - (radius * (1/2) / 10 * 9);
				x_diag_right_bis[0] = x_center + (radius * (sqrt(3)/2) / 10 * 9);

				if (output_mode == 0)
				{
					std::cout << "            Done!" << std::endl << std::endl;
				}

				Do_radius = false;

			}	   // if (Do_radius = true) close
			

		}	   // if (iter=1) close

		// If I am not in the first iteration
		else
		{

			if (output_mode == 0)
			{
				std::cout << std::endl << "          Background pixel count of the mask is: " << background_masked_count [iter-2] ;
				std::cout << std::endl << "          Foreground pixel count of the mask is: " << foreground_masked_count [iter-2] ;
			}

			// The error is calculated as the difference between foreground
			// pixels count on the mask and foreground pixels count on
			// the original image, normalized on the original count
			// (so the error as a range of 1 - 0)
			foreground_error [iter-1] = ( foreground_masked_count [iter-2] - foreground_original_count)/foreground_original_count;

			if (output_mode == 0)
			{
				std::cout << std::endl << "          Esimated segmantation error factor is: " << foreground_error[iter-1] * 100 << " %" << std::endl;
			}

			// here I instantiate 3 variables to control and determinate the
			// H step increment direction, based on epsilon change, foreground
			// error change, foregroun error sign
			int dir_a = 1;
			int dir_b = 1;
			int dir_c = 1;

			// The 4th iteration is the one after the linera prediction, when 
			// other 2 H points is needed to perform the last linear prediction.
			// The first point is the predicted one, with its real error, and the
			// second H point is the predicted one + a step increment or decrement
			// (I must move to the 0 error).
			// The increment or decrement is determinate by this 3 versors.
			if (iter == 4)
			{
				if (epsilon[iter-2] - epsilon[iter-3] < 0)
				{dir_a = -1;}
				if (foreground_error[iter-1] - foreground_error[iter-2] < 0)
				{dir_b = -1;}
				if (foreground_error[iter-1] / foreground_error[iter-2] < 0)
				{dir_c = -1;}
			}
			// Here the final direction is determinated.
			dir_epsilon = dir_a * dir_b * dir_c;

			// This statement control if the error is under a certain value, 
			// that here is +/- 10%. If is true, the algorithm automatically 
			// skips the 2nd and 3rd interations, in wich a "rough" search is
			// performed, and jumps to the "fine" search (this helps to speed up
			// the process)
			
			if (iter > 1 && iter <= 3 && foreground_error[iter-1] <= 0.01 && foreground_error[iter-1] >= -0.01)
			{
				float fv = foreground_error[iter-1];

				epsilonzeroLI = epsilon [iter-2];
				iter = 4;
				foreground_error[iter-1] = fv;
				epsilon [iter-2] = epsilonzeroLI;
				if (output_mode == 0)
				{
					std::cout << std::endl << "          The error is under 10 % → Skipping to the 'fine' search" << std::endl;
				}
			}
				

			// This statement defines the threshold for which outside values of
			// the error (greater) is goig to be considered not acceptable, and
			// the search is considered failed for this sub-volume.
			// Teh last segmentation driven by the H search is performed at iter = 5, 
			// so the last evaluation is performed at iter = 6.
			if (iter > 5)
			{
				// The fail threshold
				if ( (foreground_error[iter-1] <= -0.025 || foreground_error[iter-1] >= 0.1) && H_search_fail == false )
				{
					if (output_mode == 0)
					{
						std::cout << std::endl;
						std::cout << std::endl << "          [!] The Automatic Search has FAILED! Using safe backup H value." << std::endl;
					}
					epsilonzeroLI = 500;
					iter = 5;
					H_search_fail = true;

					// Flag for the final report
					if (vol_part == 0)
					{ 	H_search_fail_central = true;   }
					if (vol_part == 1)
					{ 	H_search_fail_bottom = true;   }
					if (vol_part == -1)
					{ 	H_search_fail_top = true;   }					
				}
				else
				{
					H_search_fail = false;
				}

				// The following statements control the sub-volume switch
				if (vol_part == 0 && iter != 0 && H_search_fail == false)
				{

					vol_part = 1;
					if (output_mode == 0)
					{
						std::cout << std::endl << "          Now scanning the bottom part of the volume ..." << std::endl;
					}
					startH_value = epsilon[iter-2];
					// Iter reset;
					iter = 0;
					// Setting the maximum intensity for this sub-volume
					maximum_intensity = maximum_intensity_b0 [(central_slice - 1) + ( vol_part * 5 )];

					// Re-calculating bins, levels and intensity increment
					histogramBins = maximum_intensity / 2;
					levelsNumber = histogramBins + 1;
					intensityIncrement = maximum_intensity / histogramBins;

					// Clearing the intensity count vector and resizing it
					intensityCount.clear();
					intensityCount.resize(levelsNumber);
				}

				if (vol_part == 1 && iter != 0 && H_search_fail == false)
				{

					vol_part = -1;

					if (output_mode == 0)
					{
						std::cout << std::endl << "          Now scanning the top part of the volume ..." << std::endl;
					}
					// Iter reset;
					iter = 0;
					// Setting the maximum intensity for this sub-volume
					maximum_intensity = maximum_intensity_b0 [(central_slice - 1) + ( vol_part * 5 )];

					// Re-calculating bins, levels and intensity increment
					histogramBins = maximum_intensity / 2;
					levelsNumber = histogramBins + 1;
					intensityIncrement = maximum_intensity / histogramBins;

					// Clearing the intensity count vector and resizing it
					intensityCount.clear();
					intensityCount.resize(levelsNumber);
				}

				// If all sub-volume as been performed, the search ends
				if (vol_part == -1 && iter != 0 && H_search_fail == false)
				{
					if (output_mode == 0)
					{
						std::cout << std::endl << "          Automatic search is completed!" << std::endl;
					}
					std::cout << "          Done!" << std::endl;
					break;
				}
				
	
			}	   // if (iter>5) close

		}		// If I am not in the first iteration close


		// Iteration 3 and 5 are the ones in which the L.I. is performed
		// ( I have 2 couple of values of sigma and error)
		if ( (iter == 3 || iter == 5)  && H_search_fail == false )
		{
			if (iter == 3)
			{
				if (output_mode == 0)
				{
					std::cout << std::endl << "          Computing 'rough' Linear Prediction: " << std::endl;
				}
			}
			else
			{
				if (output_mode == 0)
				{
					std::cout << std::endl << "          Computing 'fine' Linear Prediction: " << std::endl;
				}
			}

			// This is simply a straight line equation 
			epsilonzeroLI = epsilon[iter-2]  + 
				(  (0 - foreground_error[iter-1]) / 
				 ( (foreground_error[iter-1] - foreground_error[iter-2]) / 
				  (epsilon[iter-2] - epsilon[iter-3]) )  );

			if (output_mode == 0)
			{
				std::cout << std::endl << "          Estimated Optimal Heaviside function value is: " << epsilonzeroLI << std::endl;
			}
		}

		

		/*=========================================================================
		 * 
		 ScalarChanAndVeseSparseLevelSetImageFilter 3 dimensional
		 * 
		 =========================================================================*/

		// The Chan and vese levelset is performed for each iteration > 0,
		// because int iter = 0 the b0 images are evalueated
		if (iter != 0)
		{

		typedef itk::ImageFileWriter< InternalImageType > CnV_WriterType;
		CnV_WriterType::Pointer CnV_writer = CnV_WriterType::New();

		InternalImageType::Pointer infeatureImage = CnV_reader->GetOutput();
		// --- --- ---
		// Here again I need to fix the image origin to [0,0,0],
		// to avoid "out of requested region" issue.

		iorigin = infeatureImage->GetOrigin();
			if (output_mode == 0)
			{
				std::cout << std::endl << "     → Image origin control is now taking place..." << std::endl;
				std::cout << "          input image origin: " << iorigin << std::endl;
			}

		typedef itk::ChangeInformationImageFilter< InternalImageType > ChInfo_FilterType;
		ChInfo_FilterType::Pointer ChInfo_filter = ChInfo_FilterType::New();
			
		InternalImageType::SpacingType ext_spacing = infeatureImage->GetSpacing();

		// Fixing origin to (0,0,0) to avoid "out of largest possible region" error.
		InternalImageType::PointType origin = 0;
			if (output_mode == 0)
			{
				std::cout << "          applying new origin: " << origin << std::endl;
			}
		ChInfo_filter->SetOutputOrigin( origin );	
		ChInfo_filter->ChangeOriginOn();
		ChInfo_filter->SetInput( infeatureImage );
		ChInfo_filter->Update();
		InternalImageType::Pointer featureImage = ChInfo_filter->GetOutput();
		InternalImageType::PointType forigin = featureImage->GetOrigin();
			if (output_mode == 0)
			{
				std::cout << "          output image origin: " << forigin << std::endl;
				std::cout << "          Done!" << std::endl;
				std::cout << " " << std::endl;
			}
		// --- --- ---


		// Here I set the b0 parameters
		unsigned int nb_iteration = 25;			// default = 25
		double rms = 75;						// Convergence criteria thresold based on RMS | default = 75
		
		// Regoularization term of H (heaviside function)
		if (iter <= 2)
		{
			//  Start search value and rough step value = 500
			// for iter <= 2 is the foreground error that determinates the
			// step increment direction (and not the versors) and the amplitude
			// of the increment (> error → > increment and vice versa)
			epsilon [iter-1] = startH_value + (startH_value * (iter - 1 ) * foreground_error[iter-1] * 10) ;
		}
		// In iter 3 and 5 is where linear prediction is computed
		if (iter == 3 || iter >= 5)
		{
			// The value of the linear prediction for the fine search and for
			// the final segmentation
			epsilon [iter-1] = epsilonzeroLI;
		}
		if (iter == 4)
		{
			// In iter = 4 the step increment direction is determionated with
			// the versosr method, and the amplitude by the foregroun error
			// (> error → > increment and vice versa), but here I am in the
			// "fine" search case, so the step amlitude is lower than in the
			// "rough" search case (iter <= 3)
			epsilon [iter-1] = epsilonzeroLI + abs(foreground_error[iter-1] * 1000) * dir_epsilon;

			// If the foreground error is very little, lass than the float
			// precision (may occur) a step increment of 0 is applied, causing
			// a fail in the linear prediction (linear interpolation 
			// between two same value)
			if (epsilon [iter-1] == epsilon [iter-2])
			{
				epsilon [iter-1] = epsilon [iter-1] + 0.1 * dir_epsilon;
			}
		}

		// If H value is lower than 100. H maust be positive.
		// Empirically, values lower 100 drives very high errors in segmentation
		// and H search.
		if (epsilon [iter-1] < 100)
			{
				epsilon [iter-1] = 100;
			}

			if (output_mode == 0)
			{
				std::cout << "     → Setting up H at: " << epsilon [iter-1] << std::endl << std::endl;
			}

		double curvature_weight = 100;		// Weight of lenght(C) → Reduces boundary fragmentation (must be a very high value) | default = 100
		double area_weight = 0.;		// Weight of Area(Insight(C)) → Increases internal area (also little improve smoothness) | default = 0
		double reinitialization_weight = 0.;	// | default = 0
		double volume_weight = 0.;		// | default = 0
		double volume = 0.;			// | default = 0
		double l1 = 1.;			// lambda 1, wieght of u0 inside C | default = 1 
		double l2 = 1.;			// lambda 2, wieght of u0 outside C | default = 1
		// u0: image
		// C: any other variable curve																										


    typedef itk::ScalarChanAndVeseLevelSetFunctionData< InternalImageType,
    InternalImageType > DataHelperType;
 
    typedef itk::ConstrainedRegionBasedLevelSetFunctionSharedData<
    InternalImageType, InternalImageType, DataHelperType > SharedDataHelperType;
 
    typedef itk::ScalarChanAndVeseLevelSetFunction< InternalImageType,
    InternalImageType, SharedDataHelperType > LevelSetFunctionType;
 
    //  I declare now the type of the numerically discretized Step and Delta functions that
    //  will be used in the level-set computations for foreground and background regions
  
    typedef itk::AtanRegularizedHeavisideStepFunction< ScalarPixelType,
    ScalarPixelType >  DomainFunctionType;
 
    DomainFunctionType::Pointer domainFunction = DomainFunctionType::New();
    domainFunction->SetEpsilon( epsilon[iter-1] );
 
    //  I declare now the type of the FastMarchingImageFilter that
    //  will be used to generate the initial level set in the form of a distance
    //  map.
  
    typedef  itk::FastMarchingImageFilter< InternalImageType, InternalImageType >
    FastMarchingFilterType;
 
    FastMarchingFilterType::Pointer  fastMarching = FastMarchingFilterType::New();
 
    //  The FastMarchingImageFilter requires the user to provide a seed
    //  point from which the level set will be generated. The user can actually
    //  pass not only one seed point but a set of them. Note the the
    //  FastMarchingImageFilter is used here only as a helper in the
    //  determination of an initial level set. I could have used the
    //  \doxygen{DanielssonDistanceMapImageFilter} in the same way.
    //
    //  The seeds are passed stored in a container. The type of this
    //  container is defined as \code{NodeContainer} among the
    //  FastMarchingImageFilter traits.
    
    typedef FastMarchingFilterType::NodeContainer  NodeContainer;
    typedef FastMarchingFilterType::NodeType       NodeType;
 
    NodeContainer::Pointer seeds = NodeContainer::New();

	// I instantiate the ten seeds to be used
    InternalImageType::IndexType  seedPosition1;
	InternalImageType::IndexType  seedPosition2;
	InternalImageType::IndexType  seedPosition3;
	InternalImageType::IndexType  seedPosition4;
	InternalImageType::IndexType  seedPosition5;
	InternalImageType::IndexType  seedPosition6;
	InternalImageType::IndexType  seedPosition7;
	InternalImageType::IndexType  seedPosition8;
	InternalImageType::IndexType  seedPosition9;
	InternalImageType::IndexType  seedPosition10;
	InternalImageType::IndexType  seedPosition11;
	InternalImageType::IndexType  seedPosition12;


			if (output_mode == 0)
			{
				std::cout << "     → Setting Multiple Initialization coordinates ..." << std::endl;
			}
			
	// Here I set the coordinates for every seed in the form (x,y,z)
	// z is fixed at the central slice, so at the half of the volume.
	seedPosition1[0] = ( x_left[0] );
	seedPosition1[1] = ( y_center );
	seedPosition1[2] = ( size_z/2 );

	seedPosition2[0] = ( x_left[1] );
	seedPosition2[1] = ( y_center );
	seedPosition2[2] = ( size_z/2 );

	seedPosition3[0] = ( x_right[0] );
	seedPosition3[1] = ( y_center );
	seedPosition3[2] = ( size_z/2 );

	seedPosition4[0] = ( x_right[1] );
	seedPosition4[1] = ( y_center );
	seedPosition4[2] = ( size_z/2 );

	seedPosition5[0] = ( x_center );
	seedPosition5[1] = ( y_top[0] );
	seedPosition5[2] = ( size_z/2 );

	seedPosition6[0] = ( x_center );
	seedPosition6[1] = ( y_top[1] );
	seedPosition6[2] = ( size_z/2 );

	seedPosition7[0] = ( x_diag_left[0] );
	seedPosition7[1] = ( y_diag_left[0] );
	seedPosition7[2] = ( size_z/2 );

	seedPosition8[0] = ( x_diag_left[1] );
	seedPosition8[1] = ( y_diag_left[1] );
	seedPosition8[2] = ( size_z/2 );

	seedPosition9[0] = ( x_diag_right[0] );
	seedPosition9[1] = ( y_diag_right[0] );
	seedPosition9[2] = ( size_z/2 );

	seedPosition10[0] = ( x_diag_right[1] );
	seedPosition10[1] = ( y_diag_right[1] );
	seedPosition10[2] = ( size_z/2 );

	seedPosition11[0] = ( x_diag_right_bis[0] );
	seedPosition11[1] = ( y_diag_right_bis[0] );
	seedPosition11[2] = ( size_z/2 );

	seedPosition12[0] = ( x_diag_left_bis[0] );
	seedPosition12[1] = ( y_diag_left_bis[0] );
	seedPosition12[2] = ( size_z/2 );
			
	// Here I instantiate the seed radius in pixels.
	// I use an equal radius for every seed.
	const double initialDistance = 3;

 
    NodeType node1;
    NodeType node2;
    NodeType node3;
    NodeType node4;
    NodeType node5;
    NodeType node6;
	NodeType node7;
    NodeType node8;
	NodeType node9;
    NodeType node10;
    NodeType node11;
    NodeType node12;
 
    const double seedValue = - initialDistance;
 
    node1.SetValue( seedValue );
    node1.SetIndex( seedPosition1 );
    node2.SetValue( seedValue );
    node2.SetIndex( seedPosition2 );
    node3.SetValue( seedValue );
    node3.SetIndex( seedPosition3 );
    node4.SetValue( seedValue );
    node4.SetIndex( seedPosition4 );
    node5.SetValue( seedValue );
    node5.SetIndex( seedPosition5 );
    node6.SetValue( seedValue );
    node6.SetIndex( seedPosition6 );
    node7.SetValue( seedValue );
    node7.SetIndex( seedPosition7 );
    node8.SetValue( seedValue );
    node8.SetIndex( seedPosition8 );
    node9.SetValue( seedValue );
    node9.SetIndex( seedPosition9 );
    node10.SetValue( seedValue );
    node10.SetIndex( seedPosition10 );
    node11.SetValue( seedValue );
    node11.SetIndex( seedPosition11 );
    node12.SetValue( seedValue );
    node12.SetIndex( seedPosition12 );
 
    //  The list of nodes is initialized and then every node is inserted using
    //  the \code{InsertElement()}.
  
    seeds->Initialize();
    seeds->InsertElement( 0, node1 );
    seeds->InsertElement( 1, node2 );
    seeds->InsertElement( 2, node3 );
    seeds->InsertElement( 3, node4 );
    seeds->InsertElement( 4, node5 );
    seeds->InsertElement( 5, node6 );
	seeds->InsertElement( 6, node7 );
    seeds->InsertElement( 7, node8 );
    seeds->InsertElement( 8, node9 );
    seeds->InsertElement( 9, node10 );
    seeds->InsertElement( 10, node11 );
    seeds->InsertElement( 11, node12 );


			if (output_mode == 0)
			{
				std::cout << "          node 1: " << node1.GetIndex() << std::endl;
				std::cout << "          node 2: " << node2.GetIndex() << std::endl;
				std::cout << "          node 3: " << node3.GetIndex() << std::endl;
				std::cout << "          node 4: " << node4.GetIndex() << std::endl;
				std::cout << "          node 5: " << node5.GetIndex() << std::endl;
				std::cout << "          node 6: " << node6.GetIndex() << std::endl;
				std::cout << "          node 7: " << node7.GetIndex() << std::endl;
				std::cout << "          node 8: " << node8.GetIndex() << std::endl;
				std::cout << "          node 9: " << node9.GetIndex() << std::endl;
				std::cout << "          node 10: " << node10.GetIndex() << std::endl;
				std::cout << "          node 11: " << node11.GetIndex() << std::endl;
				std::cout << "          node 12: " << node12.GetIndex() << std::endl;
			}

 
    //  The set of seed nodes is passed now to the
    //  FastMarchingImageFilter with the method
    //  \code{SetTrialPoints()}.
  
    fastMarching->SetTrialPoints(  seeds  );
  
    //  Since the FastMarchingImageFilter is used here just as a
    //  Distance Map generator. It does not require a speed image as input.
    //  Instead the constant value $1.0$ is passed using the
    //  \code{SetSpeedConstant()} method.
  
    fastMarching->SetSpeedConstant( 1.0 );
 
    //  The FastMarchingImageFilter requires the user to specify the
    //  size of the image to be produced as output. This is done using the
    //  \code{SetOutputSize()}. Note that the size is obtained here from the
    //  output image of the smoothing filter. The size of this image is valid
    //  only after the \code{Update()} methods of this filter has been called
    //  directly or indirectly.

			if (output_mode == 0)
			{
				std::cout << std::endl << "     → Now Chan and Vese 3D Dense level Set Filter is going to be performed ..." << std::endl;

				std::cout << std::endl << "          → Performing Fast Marching ... (this will take a while) " << std::endl;
			}

    fastMarching->SetOutputSize(
      featureImage->GetBufferedRegion().GetSize() );
    fastMarching->Update();

			if (output_mode == 0)
			{
				std::cout << "            Done!" << std::endl << std::endl;

				std::cout << "          → Performing Level Set ... (this may take a while) " << std::endl;
			}
 
    //  I declare now the type of the ScalarChanAndVeseDenseLevelSetImageFilter that
    //  will be used to generate a segmentation.
   
    typedef itk::ScalarChanAndVeseDenseLevelSetImageFilter< InternalImageType,
    InternalImageType, InternalImageType, LevelSetFunctionType,
    SharedDataHelperType > MultiLevelSetType;						// Sparse <--> Dense
 
    MultiLevelSetType::Pointer levelSetFilter = MultiLevelSetType::New();
 
    //  I set the function count to 1 since a single level-set is being evolved.
  
    levelSetFilter->SetFunctionCount( 1 );
 
    //  Set the feature image and initial level-set image as output of the
    //  fast marching image filter.
  
    levelSetFilter->SetFeatureImage( featureImage );
    levelSetFilter->SetLevelSet( 0, fastMarching->GetOutput() );
 
    //  Once activiated the level set evolution will stop if the convergence
    //  criteria or if the maximum number of iterations is reached.  The
    //  convergence criteria is defined in terms of the root mean squared (RMS)
    //  change in the level set function. The evolution is said to have
    //  converged if the RMS change is below a user specified threshold.  In a
    //  real application is desirable to couple the evolution of the zero set
    //  to a visualization module allowing the user to follow the evolution of
    //  the zero set. With this feedback, the user may decide when to stop the
    //  algorithm before the zero set leaks through the regions of low gradient
    //  in the contour of the anatomical structure to be segmented.
   
    levelSetFilter->SetNumberOfIterations( nb_iteration );
    levelSetFilter->SetMaximumRMSError( rms );
 
    //  Often, in real applications, images have different pixel resolutions. In such
    //  cases, it is best to use the native spacings to compute derivatives etc rather
    //  than sampling the images.
  
    levelSetFilter->SetUseImageSpacing( 1 );
 
    //  For large images, I may want to compute the level-set over the initial supplied
    //  level-set image. This saves a lot of memory.
  
    levelSetFilter->SetInPlace( false );
 
    //  For the level set with phase 0, set different parameters and weights. This may
    //  to be set in a loop for the case of multiple level-sets evolving simultaneously.
  
    levelSetFilter->GetDifferenceFunction(0)->SetDomainFunction( domainFunction );
    levelSetFilter->GetDifferenceFunction(0)->SetCurvatureWeight( curvature_weight );
    levelSetFilter->GetDifferenceFunction(0)->SetAreaWeight( area_weight );
    levelSetFilter->GetDifferenceFunction(0)->SetReinitializationSmoothingWeight( reinitialization_weight );
    levelSetFilter->GetDifferenceFunction(0)->SetVolumeMatchingWeight( volume_weight );
    levelSetFilter->GetDifferenceFunction(0)->SetVolume( volume );
    levelSetFilter->GetDifferenceFunction(0)->SetLambda1( l1 );
    levelSetFilter->GetDifferenceFunction(0)->SetLambda2( l2 );
 
    levelSetFilter->Update();

			if (output_mode == 0)
			{
				std::cout << "            Done!" << std::endl;
			}

	if (iter == 5)
	{
		if (output_mode == 0)
		{
			std::cout << std::endl << "     Sub-Volume successfully segmented!" << std::endl;
		}
	}

	// Generating actual slice numer string, wich include the sigma value.
	// Is necessary to create different file name at each cycle, because
	// otherwise the reader does not updates itself
	std::string ev;
	std::stringstream epsilon_value;
	epsilon_value << epsilon[iter-1];
	ev = epsilon_value.str();		// actual slice number passed to "sn"

	if (vol_part == 0)
	{
		central_segmented_vol = segmented_dir;		// passing output directory to "segmented_dir"
		central_segmented_vol += "/Segmented_Volume";		// adding the new folder
		central_segmented_vol += ev;		// adding the new folder
		central_segmented_vol += "central";
		central_segmented_vol += ".nii";		// adding the new folder
		segmented_vol = central_segmented_vol;
	}
	if (vol_part == -1)
	{
		top_segmented_vol = segmented_dir;		// passing output directory to "segmented_dir"
		top_segmented_vol += "/Segmented_Volume";		// adding the new folder
		top_segmented_vol += ev;		// adding the new folder
		top_segmented_vol += "top";
		top_segmented_vol += ".nii";		// adding the new folder
		segmented_vol = top_segmented_vol;
	}
	if (vol_part == 1)
		{
			bottom_segmented_vol = segmented_dir;		// passing output directory to "segmented_dir"
			bottom_segmented_vol += "/Segmented_Volume";		// adding the new folder
			bottom_segmented_vol += ev;		// adding the new 
			bottom_segmented_vol += "bottom";
			bottom_segmented_vol += ".nii";		// adding the new folder
			segmented_vol = bottom_segmented_vol;
		}


		CnV_writer->SetFileName(segmented_vol);
		CnV_writer->SetInput(levelSetFilter->GetOutput());
		CnV_writer->Update();

			if (output_mode == 0)
			{
				std::cout << std::endl << "     Segmented volume saved as: " << segmented_vol << std::endl << std::endl;
			}



			/*==================================================================
			 
			 Clock Probe part
			 
			 =================================================================*/
			// Here I estimate the time needed to complete process all the slices

			int time;				// Defining the time variable 
			int min;				// Defining the minutes variable 
			int sec;				// Defining the seconds variable 

			// Calcoating the remaining time to process all slices
			// Stopping the clock probe function to get the time in this moment
			H_clock.Stop();							
			int counter_index;
			if (vol_part == 0)
			{ counter_index = 2; }
			if (vol_part == 1)
			{ counter_index = 1; }
			if (vol_part == -1) 
			{ counter_index = 0; }


			// I calculate the mean (of all already completed cycle) at every cycle,
			// to get a more precise estimation.
			// "clock.GetMean" gives the mean of all cycles
			time = ( H_clock.GetMean() * (( 6 * counter_index ) + (6 - iter)) );

			if (vol_part == -1 && iter >= 5)
			{ 
				time = 0;
			}

			if ( time > 60)
			{
				min = time / 60;
				sec = time - (min*60);
				if (output_mode == 0)
				{
					std::cout << std::endl << "     ~ Estimated time to process all remaining slices: " 
						<< min << " min " << sec << " sec" << std::endl << std::endl;
				}
				else
				{
					std::cout << "\r" << "  |" << std::flush;
					for (int  c = 0; c <= (iter + (abs(counter_index-2) * 6) ); c++)
					{
						std::cout << "==>|" << std::flush;

					}
					for (int  c = 0; c < (( 6 * counter_index ) + (6 - iter)  ); c++)
					{
						std::cout << "   |" << std::flush;

					}
					std::cout << "  (" 
						<< min << " min " << sec << " sec left)          " << std::flush;
				}
			}
			else
			{
				sec = time;
				if (output_mode == 0)
				{
					std::cout << std::endl << "     ~ Estimated time to process all remaining slices: " 
						<< sec << " sec" << std::endl << std::endl;
				}
				else
				{
					std::cout << "\r" << "  |" << std::flush;
					for (int  c = 0; c <= (iter + (abs(counter_index-2) * 6) ); c++)
					{
						std::cout << "==>|" << std::flush;

					}
					for (int  c = 0; c < (( 6 * counter_index ) + (6 - iter)  ); c++)
					{
						std::cout << "   |" << std::flush;

					}
					std::cout << "  (" 
						<< sec << " sec left)           " << std::flush;
				}
			}


			
		} // if iter != 0 close

		
	} // search for close
	// -- -- - SEARCH FOR END - -- --- //



	/*=========================================================================
	  
	 Post-processing
	  
	 =========================================================================*/
	// Here I perform some post-processing operations on the segmented slices.
	// So, I need to extract them from the volume.
	// 1 - Connected components
	// 2 - Hole Filling
	// 3 - Light Erosion
	// 4 - Masking the top slices with the central mask 

	// CentralMask è la maschera estratta dalla fetta centrale

	std::cout << std::endl <<"→ Now post-processing filters are going to be performed ...  [11/19]"<< std::endl;

	// A flag: I must extrat the central slice before the others
	bool first_extraction = true;

	OutputImageType::Pointer CentralMask;

	// --- -- - Slice Extraction - -- --- //
	// Here I load the segmented volume and I extract the slices
	// one by one, beginning from the central one, to the first one.

	// ----------- FOR CYCLE START ----------- //
	for ( int i = 0; i <= size_z - 1; i++  )

	{		// for cycle open

		// I must extrat the central slice before the others
		if (first_extraction == true)
		{
			i = central_slice - 1;

		}

		start[2] = i;

		// Using slicenUmberShowed instead of sliceNumber produces a more human
		// readeable output and gives coerency with slices generated file names
		// with other IRCCS study (ad-hoc fix)
		sliceNumberShowed = i + 1;
		if (output_mode == 0)
		{
			std::cout <<"     → Now processing slice "<< sliceNumberShowed << std::endl;
		}

		if (first_extraction == true)
		{
			if (output_mode == 0)
			{
				std::cout << "          → Generating mask ..." <<  std::endl;
			}
		}

		// Finally, an \doxygen{ImageRegion} object is created and initialized with
		// the start and size I just prepared using the slice information.

		InputImageType::RegionType desiredRegion;
		desiredRegion.SetSize( size );
		desiredRegion.SetIndex( start );

		// Then the region is passed to the filter using the
		// SetExtractionRegion() method.

		ExtIm_filter->SetExtractionRegion( desiredRegion );
		// I extract the segmented slices from the levelset, previously saved
		CnV_ReaderType::Pointer pp_reader = CnV_ReaderType::New();

		if ( i <= (central_slice - 1) - 2 )
		{
			segmented_vol = bottom_segmented_vol;
		}
		else if ( i >= (central_slice - 1) + 2 )
		{
			segmented_vol = top_segmented_vol;
		}
		else
		{
			segmented_vol = central_segmented_vol;
		}

		if (output_mode == 0)
		{
			std::cout <<"          → Loading: "<< segmented_vol << std::endl;
		}

		pp_reader -> SetFileName(segmented_vol);	
		ExtIm_filter->SetInput( pp_reader->GetOutput() );
		// and I save them as "extractedImage"
		OutputImageType::Pointer extractedImage = ExtIm_filter->GetOutput();

		

		// --- -- - Connected Components - -- --- //
		// Here I proceed to mantain only the largest mask in case I have some
		// scattered and sparse pieces of mask

		// Short pixel type is required by ConnectedComponentsFilter becouse
		// an integral pixel type is needed.
		typedef  short shortPixelType;
		typedef itk::Image< shortPixelType, 2 >  shortImageType;
		
		typedef itk::ConnectedComponentImageFilter <OutputImageType, shortImageType >
			ConnectedComponentImageFilterType;
		ConnectedComponentImageFilterType::Pointer connected = 
			ConnectedComponentImageFilterType::New ();
		connected->SetInput(extractedImage);
		connected->Update();
		if (output_mode == 0)
		{
			std::cout << "          → Connected Components filter operations is going to be performed ..." <<  std::endl;
			std::cout << "            Number of objects: " << connected->GetObjectCount() << std::endl;
		}
		typedef itk::LabelShapeKeepNObjectsImageFilter< shortImageType 
			> LabelShapeKeepNObjectsImageFilterType;
		LabelShapeKeepNObjectsImageFilterType::Pointer 
			labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
		labelShapeKeepNObjectsImageFilter->SetInput( connected->GetOutput() );
		labelShapeKeepNObjectsImageFilter->SetBackgroundValue( 0 );
		// To mantain only the (one) main mask
		labelShapeKeepNObjectsImageFilter->SetNumberOfObjects( 1 );
		// This is the critera to wich mask mantain. This is the maximum number
		// of pixels.
		labelShapeKeepNObjectsImageFilter->
			SetAttribute( LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
		typedef itk::RescaleIntensityImageFilter< shortImageType, OutputImageType
			> RescaleFilterType;
		RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
		rescaleFilter->SetOutputMinimum(0);
		rescaleFilter->SetOutputMaximum(1);
		rescaleFilter->SetInput(labelShapeKeepNObjectsImageFilter->GetOutput());
		if (output_mode == 0)
		{
			std::cout << "            Done!" <<  std::endl;
		}



		// --- -- - Hole Filling - -- --- //
		// Here I fill holes in the mask

		if (output_mode == 0)
		{
			std::cout << "          → Hole Filling filter operations is going to be performed ..." <<  std::endl;
		}
		typedef itk::BinaryFillholeImageFilter <OutputImageType>
			BinaryFillholeImageFilterType;
		BinaryFillholeImageFilterType::Pointer fillhole = 
			BinaryFillholeImageFilterType::New ();
		fillhole->SetInput(rescaleFilter->GetOutput());
		fillhole->SetForegroundValue(1);
		fillhole->Update();
		if (output_mode == 0)
		{
			std::cout << "            Done!" <<  std::endl;
		}
		
		

		// --- -- - Erosion - -- --- //
		// I perform erosion to remove some noise from the image

		if (output_mode == 0)
		{
			std::cout << "          → Erosion filter operations is going to be performed ..." <<  std::endl;
		}

		unsigned int radius = 2;

		typedef itk::BinaryBallStructuringElement<
			OutputImageType::PixelType, 2> StructuringElementType;
		StructuringElementType structuringElement;
		structuringElement.SetRadius(radius);
		structuringElement.CreateStructuringElement();

		typedef itk::BinaryErodeImageFilter <OutputImageType, OutputImageType, StructuringElementType>
			BinaryErodeImageFilterType;

		BinaryErodeImageFilterType::Pointer erodeFilter
			= BinaryErodeImageFilterType::New();
		erodeFilter->SetInput(fillhole->GetOutput());
		erodeFilter->SetKernel(structuringElement);

		if (output_mode == 0)
		{
			std::cout << "            Done!" <<  std::endl;
		}



		// Condition: if the extracted slice is the central slice,
		// this is the slice used to create the mask to be applayed
		// to the other first slices (from 0 to central - 1).
		// The gola is to avoid cases in which the levelset filter
		// diverges and segment the bladder.
		// With this method I ensure that the resulting segmented
		// lower slices as a segmented surface not larger than the central one.
		// So I try to limit the bladder segmentation.

		//Here I extract the mask from the central slice

		// workaround!!! (how to a pointer out of a statement?)
		std::string CentralMask_dir;
		CentralMask_dir = tmp_dir;		
		CentralMask_dir += "/CentralMask.nii";	
		//////!!

		if (first_extraction == true)
		{

			OutputImageType::Pointer CentralMask = erodeFilter->GetOutput();
			// workaround!!!
			typedef itk::ImageFileWriter< OutputImageType > WriterType;
			WriterType::Pointer writer = WriterType::New();
			writer->SetFileName(CentralMask_dir);
			writer->SetInput(CentralMask);
			writer->Update();
			//////!!

			// I set the following variables to continue the extraction
			// from slice 0 to slice (size_z - 1)
			i = -1;
			first_extraction = false;

		}


		// If I are not in the case of the central slice mask extraction,
		// I applay the mask on the all slices

		else
		{

			OutputImageType::Pointer ErodedImage = erodeFilter->GetOutput();

			// workaround!!!
			typedef itk::ImageFileReader< OutputImageType > ReaderType;
			ReaderType::Pointer reader = ReaderType::New();
			reader->SetFileName(CentralMask_dir);
			reader->Update();
			OutputImageType::Pointer CentralMask = reader->GetOutput();
			/////////!!



			// --- -- - Masking - -- --- //

				if (output_mode == 0)
			{
				std::cout << "          → Masking filter operations is going to be performed ..." <<  std::endl;
			}

			typedef itk::MaskImageFilter< OutputImageType, OutputImageType > MaskFilterType;

			MaskFilterType::Pointer maskFilter2 = MaskFilterType::New();

			maskFilter2->SetInput(ErodedImage);
			maskFilter2->SetMaskImage(CentralMask);

			OutputImageType::Pointer MaskedImage = maskFilter2->GetOutput();

			OutputImageType::Pointer inputImage = MaskedImage;

			if (output_mode == 0)
			{
				std::cout << "            Done!" <<  std::endl <<  std::endl;
			}


			// Here I save the created masked image.

			// I instantiate reader and writer types in the following lines.

			typedef itk::ImageFileReader< OutputImageType > ReaderType;
			ReaderType::Pointer CnV_reader = ReaderType::New();

			// I need to save any slice in DCM format. Levelset filter needs floating type image, but is not supported in DCM (and GDCM),
			// so a *casting* operation is needed.
			// Here I define the conversion format -short- and its writer

			// Generating single slice output file name, usign the output directory given.
			// I use the cropped image becouse in this way I leave out non usefull region.

			// Generating actual slice numer string
			std::string sn;
			std::stringstream out_slice;
			out_slice << sliceNumberShowed;
			sn = out_slice.str();		// actual slice number passed to "sn"

			// Adding other parts of file name, so I have: tmp_dir_slice = tmp_dir + "/slice" + s + ".nii"
			std::string tmp_dir_slice;
			tmp_dir_slice += postprocessing_dir;
			tmp_dir_slice += "/slice_pp";
			if ( sliceNumberShowed < 10 )
			{
				tmp_dir_slice += "0";		// I want a name generation like this: slice01, slice02, ... , slice10, slice11, ...
			}
			tmp_dir_slice += sn;
			tmp_dir_slice += ".dcm";		// output slice format. dcm requires a casting operation, becouse float is not supported.

			if (output_mode == 0)
			{
				std::cout << "          → Post-processed image saved as: " << tmp_dir_slice << std::endl;
			}

			try
			{

				// I need to save any slice in DCM format. Levelset filter needs floating type image, but is not supported in DCM (and GDCM),
				// so a *casting* operation is needed.
				// Here casting operation takes palce, from float (levelset filter) to short (dcm).

				typedef itk::Image<float, 2>  FloatImageType;
				typedef itk::Image<short, 2>  ShortImageType;
				typedef itk::CastImageFilter< FloatImageType, ShortImageType > CastFilterType;
				CastFilterType::Pointer castFilter = CastFilterType::New();
				castFilter->SetInput(MaskedImage);

				castFilter->Update();

				typedef itk::ImageFileWriter< ShortImageType > WriterType;
				WriterType::Pointer Mask_writer = WriterType::New();

				Mask_writer->SetFileName( tmp_dir_slice );

				Mask_writer->SetInput( castFilter->GetOutput() );

				Mask_writer->UpdateLargestPossibleRegion();

				if (output_mode == 0)
				{
					std::cout << "            Done!" <<  std::endl;
					std::cout << " " << std::endl;
				}
			}
			catch( itk::ExceptionObject & excep )
			{
				std::cerr << "Exception caught !" << std::endl;
				std::cerr << excep << std::endl;
				return -1;
			} 

			
			if (output_mode == 1)
			{
				std::cout << "\r" << "  |" << std::flush;
				for (int  c = 0; c <= i; c++)
				{
					std::cout << "==>|" << std::flush;

				}
				for (int  c = 0; c < size_z - 1 - i; c++)
				{
					std::cout << "   |" << std::flush;

				}
				std::cout << "  (" 
					<< (int)((float)i/((float)size_z-1)*100) << " %)           " << std::flush;
			}
						

		} // else close

		
	}		// for cycle close
	//---------------------- EXTRACT SLICES FOR END ----------------------//

	std::cout << "            Done!" <<  std::endl;




	/*=========================================================================
	  
	 Superior Periferical Coil Tissue Segmentation
	  
	 =========================================================================*/

	// Here the aim is to try to segment the tissue around the superior part of
	// the coil (i.e. the tissue between the coil and the prostate).
	// Due to similar intensity value, Chan and Vese Level Set iclude this tissue
	// in the segmentation, but it is not prostatic tissue.
	// Moreover, this portion will result in a high false positive rate in a
	// MRI tumor search.
	// So here I implement a way to segment this portion of the image,
	// and then I subtract it to the prostate mask previously find with
	// Chan and Vese Level Set.

	// This kind of segmentation is applaied to T2 images, because diffusion
	// registrated images are too much distort in this part of the image (due 
	// to registration).

	// I start to set the T2 slices directory.
	// I expect the following directory format:
	// " 'core_dir'/T2/dcm "
	// where 'core_dir' was: "patient/patientnumber"

	std::string T2_dir;
	T2_dir = core_dir.str();		
	T2_dir += "/T2/dcm";

	// Generating T2 volume output file name, using the output directory given
	std::string T2_volume;
	T2_volume = core_dir.str();		
	T2_volume += "/Volume_Original_T2.nii"; 


	// --- -- - dcm2nii_T2 - -- --- //
	// Now I have to built the T2 volume, to make sure to have the correct order
	// of the slices.

	std::cout << std::endl << "→ Now Segmentaton of coil superior periferical tissue is going to be performed ...  [12/19]" <<  std::endl;

	if (output_mode == 0)
	{
		std::cout << "     → Building T2 volume ..." << std::endl <<  std::endl;
	}

	ReaderType::Pointer d2nT2_reader = ReaderType::New();

	d2nT2_reader->SetImageIO( dicomIO );

	nameGenerator->SetUseSeriesDetails( false );
	nameGenerator->AddSeriesRestriction("0020|9057");/*0008|0021*/
	nameGenerator->AddSeriesRestriction("0020|1041");
	nameGenerator->AddSeriesRestriction("0020|0012");
	nameGenerator->SetDirectory( T2_dir );

	try
	{
		if (output_mode == 0)
		{
			std::cout << std::endl << "       Analizing T2 directory: " << T2_dir << std::endl;
			std::cout << "       Found following DICOM Series: ";
			std::cout << std::endl;
		}

		typedef std::vector< std::string >    SeriesIdContainer;
		typedef std::vector< std::string >   FileNamesContainer;
		FileNamesContainer fileNames;

		const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

		SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
		SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
		while( seriesItr != seriesEnd )
		{
			if (output_mode == 0)
			{
				std::cout << "          " << seriesItr->c_str() << std::endl;
			}
			seriesItr++;
		}		

		std::string seriesIdentifier;
		seriesIdentifier = seriesUID.begin()->c_str();

		fileNames = nameGenerator->GetFileNames( seriesIdentifier );

		d2nT2_reader->SetFileNames( fileNames );

		try
		{
			d2nT2_reader->Update();
		}
		catch (itk::ExceptionObject &ex)
		{
			std::cout << ex << std::endl;
			return EXIT_FAILURE;
		}

		typedef itk::OrientImageFilter<ImageType,ImageType> OrienterType;
		OrienterType::Pointer orienter = OrienterType::New();

		orienter->UseImageDirectionOn();
		orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
		orienter->SetInput(d2nT2_reader->GetOutput());

		WriterType::Pointer d2nT2_writer = WriterType::New();

		d2nT2_writer->SetFileName( T2_volume );

		d2nT2_writer->SetInput( orienter->GetOutput() );

		if (output_mode == 0)
		{
			std::cout << std::endl << "       Generating the new volume ... " << std::endl;
			std::cout << "       New volume saved as: " << std::endl;
			std::cout << "       " << T2_volume << std::endl;
			std::cout << "       Done!" << std::endl << std::endl;
		}

		try
		{
			d2nT2_writer->Update();
		}
		catch (itk::ExceptionObject &ex)
		{
			std::cout << ex << std::endl;
			return EXIT_FAILURE;
		}
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}



	// --- -- - Slice Exctraction Instantation - -- --- //	

	// here I instantiate some variables for a successive slice extraction

	FilterType::Pointer T2_ExtIm_filter = FilterType::New();
	T2_ExtIm_filter->InPlaceOff();
	T2_ExtIm_filter->SetDirectionCollapseToSubmatrix();
	typedef itk::ImageFileReader< InternalImageType > T2_ReaderType;
	T2_ReaderType::Pointer T2_ExtIm_reader = T2_ReaderType::New();
	T2_ExtIm_reader->SetFileName(T2_volume);
	T2_ExtIm_reader->Update();



	// --- -- - Shrink T2 images - -- --- //

	// Here, if necessary, I proceed to subsample the T2 images, to match
	// the b0 dimensions. I assume T2 mut be greater or equal as b0
	// dimensions.
	// I perform this opreation in 3D for faster performances.

	typedef itk::ShrinkImageFilter <InputImageType, InputImageType>
		T2_ShrinkImageFilterType;

	T2_ShrinkImageFilterType::Pointer T2_shrinkFilter
		= T2_ShrinkImageFilterType::New();
	T2_shrinkFilter->SetInput(T2_ExtIm_reader->GetOutput());


	int T2_shrink_value = 1;
	InternalImageType::RegionType T2wrong_inputRegion =
		T2_ExtIm_reader->GetOutput()->GetLargestPossibleRegion();
	InternalImageType::SizeType T2wrong_size = T2wrong_inputRegion.GetSize();
	if (T2wrong_size[0] > size[0])
	{
		if (output_mode == 0)
		{
			std::cout << std::endl << "     → Fixing T2 image dimensions " << std::endl;
		}
		T2_shrink_value = T2wrong_size[0] / size[0];
		if (output_mode == 0)
		{
			std::cout << std::endl << "       T2 image dimensions now is " <<
				T2wrong_size[0] / T2_shrink_value << std::endl << std::endl;
		}
	}

	// Shrink the first dimension by a factor of T2_shrink_value 
	T2_shrinkFilter->SetShrinkFactor(0, T2_shrink_value);
	// Shrink the second dimension by a factor of T2_shrink_value 
	T2_shrinkFilter->SetShrinkFactor(1, T2_shrink_value);



	// ** ** **     ** ** **     ** ** **     
	// Now the Segmentation of Superior part of the Coil has place.
	// Since mehods to find only the tissue I want to segment, like
	// IsolatedConnected Filter, have failed on T2, I follow a completly
	// different aproach.
	//
	// Method:
	// Now the goal is eliminate the tisseue between the coil and the prostate,
	// exacty what I want to segment.
	// If I achieve this, the coil hole becomes a whole bigger hole,
	// expanding a little on its top.
	// At this point I are able to segment the whole hole obtained, that now
	// includes the coil hole and the tissue I want to remove from the prostate
	// mask.
	//
	// I follow these steps:
	//
	// Light 3d Erosion  →  Medium/Heavy 2D Open  →  Light Smoothing  →     [*1]
	//  →  Region Growing (threshold based) inside the coil  →              [*2]
	//  →  Medium/Heavy Dilatation  →  Light Gaussian Smoothing             [*3]
	//
	// [*1] Pre-processing
	// [*2] Segmentation
	// [*3] Post-processing
	// ** ** **     ** ** **     ** ** **     
	


	// --- -- - 3D Erosion - -- --- //

	// With a light 3D erosion I open the black part of the image.
	// Since the tissue I want to remove is a the bottom of a black region that
	// is well expanded in the z direction, the erosion along this axes
	// ensure a good and stable open of this area.

	if (output_mode == 0)
	{
		std::cout << "     → Performing 3D erosion " << std::endl;
	}

	typedef itk::BinaryBallStructuringElement<
		InputImageType::PixelType,
	3>                  StructuringElementType;
	StructuringElementType structuringElement;
	structuringElement.SetRadius(0.5);
	structuringElement.CreateStructuringElement();

	typedef itk::GrayscaleErodeImageFilter <InputImageType, InputImageType,
		StructuringElementType>	GrayscaleErodeImageFilterType3D;

	GrayscaleErodeImageFilterType3D::Pointer T2erodeFilter3D
		= GrayscaleErodeImageFilterType3D::New();
	T2erodeFilter3D->SetKernel(structuringElement);
	T2erodeFilter3D->SetInput(T2_shrinkFilter->GetOutput());

	if (output_mode == 0)
	{
		std::cout << "       Done! " << std::endl << std::endl;
	}

	
	
	// --- -- - Change Image Information - -- --- //
	
	// Here I proceed to control spacing and origin of the T2 volume

	// Now some control precesses are needed:
	// Origin must be [0,0,0] (to avoid "requested region out of laregest possible region" issue)
	// T2 Spacing (aka PixelResolutin) must be equal to the b0 MRI.


	InputImageType::Pointer T2_inputImage = T2erodeFilter3D->GetOutput();

	InputImageType::PointType T2_iorigin = T2_inputImage->GetOrigin();
	if (output_mode == 0)
	{
		std::cout << "     → Image origin control now is taking place..." << std::endl;
		std::cout << "       input image origin: " << iorigin << std::endl;
	}

	typedef itk::ChangeInformationImageFilter< InputImageType > ChInfo_FilterType;
	ChInfo_FilterType::Pointer T2_ChInfo_filter = ChInfo_FilterType::New();
	InputImageType::SpacingType T2_ext_spacing = T2_inputImage->GetSpacing();

	// Fixing origin to (0,0,0) to avoid "out of largest possible region" error.
	InputImageType::PointType T2_origin = 0;
	if (output_mode == 0)
	{
		std::cout << "       applying new origin: " << T2_origin << std::endl;
	}
	T2_ChInfo_filter->SetOutputOrigin( T2_origin );	
	T2_ChInfo_filter->ChangeOriginOn();
	T2_ChInfo_filter->SetInput( T2_inputImage );

	InputImageType::Pointer T2_featureImage = T2_ChInfo_filter->GetOutput();
	InputImageType::PointType T2_forigin = T2_featureImage->GetOrigin();
	if (output_mode == 0)
	{
		std::cout << "       output image origin: " << T2_forigin << std::endl;
		std::cout << "       Done!" << std::endl;
		std::cout << " " << std::endl;
	}

	// Fixing Spacing i.e. Pixel resolution issue. The ExtractImageFilter automatic changes the original spacing to (1,1),
	// but Pixel resolution as a phisical meaning!
	// Original Pixel Resolution is needed to make Level Set filter work properly.
	if (output_mode == 0)
	{
		std::cout << "     → Fixing spacing (pixel resolution)" << std::endl;
	}
	T2_ChInfo_filter->SetOutputSpacing( 1 );
	T2_ChInfo_filter->ChangeSpacingOn();
	InputImageType::SpacingType T2_new_spacing = T2_featureImage->GetSpacing();
	if (output_mode == 0)
	{
		std::cout << "       Wrong spacing is: "<< T2_ext_spacing << std::endl;
		std::cout << "       New correct spacing is: "<< T2_new_spacing << std::endl;
		std::cout << "       Fixed!"<< std::endl<< std::endl;
	}

	T2_ChInfo_filter->UpdateLargestPossibleRegion();


	

	// --- -- - Slice Extraction - -- --- //

	// Now I nedd to perform 2D filtering, so T2 volume slice extraction is 
	// needed.

	InputImageType::RegionType T2_inputRegion = T2_featureImage->GetLargestPossibleRegion();
	InputImageType::SizeType T2_size = T2_inputRegion.GetSize();
	// Collapsing z direction
	T2_size[2]=0;
	InputImageType::IndexType T2_start = T2_inputRegion.GetIndex();

	bool T2_first_extraction = true;

	// A flag to report a warning at the end of the code
	bool ThRegionGrowingFailed = false;


	// ----------- FOR CYCLE START ----------- //
	for ( int i = 0; i <= size_z - 1; i++  )

	{		// for cycle open

		// I must extrat the central slice before the others
		if (T2_first_extraction == true)
		{
			i = central_slice - 1;

		}

		// Starting slice extraction
		T2_start[2] = i;

		// Using slicenNumberShowed instead of sliceNumber produces a more human
		// readeable output and gives coerency with slices generated file names
		// with other IRCCS study (ad-hoc fix)
		sliceNumberShowed = i + 1;
		
		if (output_mode == 0)
		{
			std::cout <<"     → Now processing slice "<< sliceNumberShowed << std::endl;
		}

		// Finally, an \doxygen{ImageRegion} object is created and initialized with
		// the start and size I just prepared using the slice information.

		InputImageType::RegionType T2_desiredRegion;
		T2_desiredRegion.SetSize( T2_size );
		T2_desiredRegion.SetIndex( T2_start );

		// Then the region is passed to the filter using the
		// SetExtractionRegion() method.

		T2_ExtIm_filter->SetExtractionRegion( T2_desiredRegion );

		// I extract the segmente slices from the levelset
		T2_ExtIm_filter->SetInput( T2_featureImage );
		T2_ExtIm_filter->UpdateLargestPossibleRegion();

		// I need to extract the central slice first becouse I need
		// the coordinate to the center of the coil, to be used as seed of
		// the region growing.
		// To find it I use Houg Transform.
		// Setting the right radius parameters of the coil I ensure to
		// find the coil.
		if (T2_first_extraction == true)
		{
			if (output_mode == 0)
			{
				std::cout << std::endl <<"          → Searching for Coil center coordinates..."<< std::endl;
			}

			double NumberOfCircles;
			HoughImageType::Pointer localImage;

			// Here I define the number of circle to be find,
			// and I link the pointer to the correct image for each case. 

			NumberOfCircles = 1;
			localImage =     T2_ExtIm_filter -> GetOutput();

			// Here I set the others non variable hough parameters
			double MinimumRadius = 35;
			double MaximumRadius = 45;
			double SweepAngle = 0.4; 		// default = 0
			//double SigmaGradient = 1;		// default = 1
			//double Variance = 5;		// deafult = 5;
			//double DiscRadiusRatio = 10;	// deafult = 10;

			houghFilter->SetInput( localImage );

			houghFilter->SetNumberOfCircles( NumberOfCircles );
			houghFilter->SetMinimumRadius(   MinimumRadius );
			houghFilter->SetMaximumRadius(   MaximumRadius );
			houghFilter->SetSweepAngle( SweepAngle );
			//houghFilter->SetSigmaGradient( SigmaGradient );
			//houghFilter->SetVariance( Variance );
			//houghFilter->SetDiscRadiusRatio( DiscRadiusRatio );

			houghFilter->Update();
			OutputImageType::Pointer localAccumulator = houghFilter->GetOutput();   

			//  I can also get the circles as \doxygen{EllipseSpatialObject}. The
			//  \code{GetCircles()} function return a list of those.

			HoughTransformFilterType::CirclesListType circles;
			circles = houghFilter->GetCircles( NumberOfCircles );

			//  I can then allocate an image to draw the resulting circles as binary
			//  objects.

			typedef    short   HoughOutputPixelType;
			typedef  itk::Image< HoughOutputPixelType, 2 > HoughOutputImageType;  

			HoughOutputImageType::Pointer  localOutputImage = HoughOutputImageType::New();

			HoughOutputImageType::RegionType region;
			region.SetSize(localImage->GetLargestPossibleRegion().GetSize());
			region.SetIndex(localImage->GetLargestPossibleRegion().GetIndex());
			localOutputImage->SetRegions( region );
			localOutputImage->SetOrigin(localImage->GetOrigin());
			localOutputImage->SetSpacing(localImage->GetSpacing());
			localOutputImage->Allocate();
			localOutputImage->FillBuffer(0);

			//  I iterate through the list of circles and I draw them.

			typedef HoughTransformFilterType::CirclesListType CirclesListType;
			CirclesListType::const_iterator itCircles = circles.begin();

			// Here I get the (x,y) center coordinates and radius
			x_center = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
			y_center =  (*itCircles)->GetObjectToParentTransform()->GetOffset()[1]; 
			radius = (*itCircles)->GetRadius()[0];

			if (output_mode == 0)
			{
				std::cout << "            Coil (x) coordinate is: " << x_center << std::endl;
				std::cout << "            Coil (y) coordinate is: " << y_center << std::endl;
				std::cout << "            Done!" << std::endl << std::endl;
			}


			
			// I set the following variables to continue the extraction
			// from slice 0 to slice (size_z - 1)
			i = -1;
			T2_first_extraction = false;

		}
		

		// If I am not in the case of the central slice

		else
		{
			
			// Here I get the maximum intensity value of each T2 images,
			// used to set the threshold of the region growin filter.
			typedef itk::StatisticsImageFilter<OutputImageType> Coil_StatisticsImageFilterType;
			Coil_StatisticsImageFilterType::Pointer Coil_statisticsImageFilter
				= Coil_StatisticsImageFilterType::New ();
			Coil_statisticsImageFilter->SetInput(T2_ExtIm_filter->GetOutput());
			Coil_statisticsImageFilter->UpdateLargestPossibleRegion();
			float maximum_intensity_Coil = Coil_statisticsImageFilter->GetMaximum();
			if (output_mode == 0)
			{
				std::cout << std::endl << "          → Setting threshold ..." << std::endl;
				std::cout << std::endl << "            Maximum intensity is: " << maximum_intensity_Coil << std::endl;
			}
			// This is the threshold, relative to the maximum. Values from 0 to 
			// this one will be included in the mask.
			// The multiplication factor is to take the lower intensity values
			// of the T2 slices, supposed to be the background.
			float Coil_threshold = maximum_intensity_Coil * 0.03;
			if (output_mode == 0)
			{
				std::cout << std::endl << "            Threshold is: " << Coil_threshold << std::endl;
			}


			// --- -- - Open - -- --- //

			// A medium/hard open filter to open the black region to obtain
			// one whole large hole by the coil.

			if (output_mode == 0)
			{
				std::cout << std::endl <<"          → Performing Open ..."<< std::endl;
			}

			// - Erosion -

			typedef itk::BinaryBallStructuringElement<
				OutputImageType::PixelType,
			2>                  StructuringElementType;
			StructuringElementType T2structuringElement;
			T2structuringElement.SetRadius(3);
			T2structuringElement.CreateStructuringElement();

			typedef itk::GrayscaleErodeImageFilter <OutputImageType, OutputImageType, StructuringElementType>
				GrayscaleErodeImageFilterType;

			GrayscaleErodeImageFilterType::Pointer T2erodeFilter
				= GrayscaleErodeImageFilterType::New();
			T2erodeFilter->SetInput(T2_ExtIm_filter->GetOutput());
			T2erodeFilter->SetKernel(T2structuringElement);

			// - Dilatation -

			typedef itk::GrayscaleDilateImageFilter <OutputImageType, OutputImageType, StructuringElementType>
				GrayscaleDilateImageFilterType;

			GrayscaleDilateImageFilterType::Pointer T2dilateFilter
				= GrayscaleDilateImageFilterType::New();
			T2dilateFilter->SetInput(T2erodeFilter->GetOutput());
			T2dilateFilter->SetKernel(T2structuringElement);

			if (output_mode == 0)
			{
				std::cout << "            Done!"<< std::endl;
			}

			

			// --- -- - Smoothing - -- --- //

			// A Light smoothing filter.
			// I find mean filter works well.

			if (output_mode == 0)
			{
				std::cout << std::endl <<"          → Performing Mean Smoothing ..."<< std::endl;
			}

			typedef itk::MeanImageFilter<
				OutputImageType, OutputImageType >  MeanFilterType;
			MeanFilterType::Pointer meanFilter = MeanFilterType::New();
			OutputImageType::SizeType meanFilter_indexRadius;
			meanFilter_indexRadius[0] = 2; // radius along x
			meanFilter_indexRadius[1] = 2; // radius along y
			meanFilter->SetRadius( meanFilter_indexRadius );
			meanFilter->SetInput( T2dilateFilter->GetOutput() );
			meanFilter->Update();

			if (output_mode == 0)
			{
				std::cout << "            Done!"<< std::endl;
			}
			

			// --- -- - Region Growing - -- --- //

			// Segmentatoin of the new region obtained is achieved by
			// Threshold Region Growing.
			// To avoid ad-hoc solutions the threshold is setted as a percentage
			// the maximum intensity peak of each slice.

			if (output_mode == 0)
			{
				std::cout << std::endl <<"          → Performing Threshold Region Growing Segmentation with Automatic Threshold Control Algorithm ..."<< std::endl;
			}

			typedef itk::ConnectedThresholdImageFilter< OutputImageType,
			OutputImageType > ConnectedFilterType;

			ConnectedFilterType::Pointer connectedThreshold = ConnectedFilterType::New();

			// This is a flag used to control the over-segmentation
			bool Exploded = true;
			// The count of the pixel in the segmented mask
			int pixelCount = 0;
			// The replace value of the mask
			int replaceValue = 1;

			while (Exploded == true)
			{

				connectedThreshold->SetLower(  0  );
				connectedThreshold->SetUpper(  Coil_threshold  );
				connectedThreshold->SetReplaceValue( replaceValue );

				OutputImageType::IndexType  connectedThreshold_index;
				// Here I use the x and y coordinates found with Hough.
				connectedThreshold_index[0] = x_center;
				connectedThreshold_index[1] = y_center;
				connectedThreshold->SetSeed( connectedThreshold_index );

				connectedThreshold->SetConnectivity( ConnectedFilterType::FullConnectivity );

				connectedThreshold->SetInput( meanFilter->GetOutput() );
				connectedThreshold->Update();


				// Here I implement a method to verify that the find threshold
				// doesn't cause an overstimated segmentation (in certain cases
				// the mask "explodes" an segment quite all the image)

				// First of all, I scan every pixel of every slice
				for (int x = 0; x <= T2_size[0]; x++) 
				{
					for (int y = 0; y <= T2_size[1]; y++)
					{
						// Then, I get the pixels values
						OutputImageType::IndexType pixelIndex;
						pixelIndex[0] = x;
						pixelIndex[1] = y;
						OutputImageType::PixelType pixelValue;
						pixelValue = connectedThreshold->GetOutput()->GetPixel(pixelIndex);
						if (pixelValue == 1)
						{
							// Here I count the pixels of the mask
							pixelCount++;

							if (pixelIndex[1] <= T2_size[1] / 2 )
							{
								// One control criteria is to not allow
								// the mask to expand above the first half
								// of the image, where the prostate is present
								// If I am in this case, I simply set a count
								// of pixel not allowed in the next control
								// criteria
								pixelCount = T2_size[0] * T2_size[1];
								break;
							}
						}


					}		// y close
				}		// x close

				// The second control criteria is that the mask must have a 
				// limited surface. This limit is relative to the slice dimension.
				if (pixelCount <= (T2_size[0] * T2_size[1] / 10) )
				{
					// If the limit is respected, all the two controls is passed,
					// the flag "Exploded" setted to false, as conseguence  the
					// while loop in breaked
					Exploded = false;
				}
				else
				{
					// If the control is not passed, the threshold is lowered,
					// and the loop repeated
					Coil_threshold--;
					pixelCount = 0;

					if (Coil_threshold <= 10)
					{
						// In this case the threshold search is considered failed,
						// and a mask of 0 (no mask) is created
						replaceValue = 0;
						std::cout << std::endl <<"          [!] WARNING: Threshold automatic search failed!"<< std::endl;
						std::cout <<"                       Coil tissue is not removed in this slice."<< std::endl;
						ThRegionGrowingFailed = true;

					}

				}

			}		// while close

			if (output_mode == 0)
			{
				std::cout << "            Done!"<< std::endl;
			}

			

			// --- -- - Dilatation - -- --- //
			
			// Now Dilatation and Gaussian smoothing are performed in ordeer to
			// abtain a more smooth profile of the segmented mask obtained
			// from the region growing.
			// Since gaussian smooting highly erode the mask,
			// I had to dilatate it first.

			if (output_mode == 0)
			{
				std::cout << std::endl <<"          → Performing Post-processing ..."<< std::endl;
			}
			
			GrayscaleDilateImageFilterType::Pointer T2dilateFilter2
				= GrayscaleDilateImageFilterType::New();
			T2structuringElement.SetRadius(5);
			T2dilateFilter2->SetKernel(T2structuringElement);
			T2dilateFilter2->SetInput(connectedThreshold->GetOutput());

			

			// --- -- - Gaussian Smoothing - -- --- //

			typedef itk::DiscreteGaussianImageFilter<
				OutputImageType, OutputImageType >  T2GaussianFilterType;

			// Create and setup a Gaussian filter
			T2GaussianFilterType::Pointer T2gaussianFilter = T2GaussianFilterType::New();
			T2gaussianFilter->SetInput( T2dilateFilter2->GetOutput() );
			T2gaussianFilter->SetVariance(1);

			if (output_mode == 0)
			{
				std::cout << "            Done!"<< std::endl;
			}
			


			// --- -- - Slice Writer - -- --- //

			// To save ad dicom format, is necessary to convert float type
			// to short type.

			try
			{
				typedef short ShortPixelType;
				typedef itk::Image< ShortPixelType, 2 > ShortImageType;
				typedef itk::CastImageFilter<
					OutputImageType,
				ShortImageType > CastFilterType;
				typedef itk::ImageFileWriter< ShortImageType > WriterType;
				WriterType::Pointer writer_Coil = WriterType::New();
				WriterType::Pointer writer_T2sub = WriterType::New();
				
				CastFilterType::Pointer caster = CastFilterType::New();
				CastFilterType::Pointer casterT2sub = CastFilterType::New();


				// Generating single slice output file name, usign the output directory given.

				std::string sn;
				std::stringstream out_slice;
				out_slice << sliceNumberShowed;
				sn = out_slice.str();		// actual slice number passed to "sn"
				// Adding other parts of file name, so I have: tmp_dir_slice = tmp_dir + "/slice" + s + ".nii"
				std::string T2_slice;
				std::string CoilMask_slice;
				T2_slice += postprocessing_dir;
				CoilMask_slice += postprocessing_dir;
				T2_slice += "/T2sub_slice";
				CoilMask_slice += "/CoilMask_slice";
				if ( sliceNumberShowed < 10 )
				{
					T2_slice += "0";
					CoilMask_slice += "0";
				}
				T2_slice += sn;
				CoilMask_slice += sn;
				T2_slice += ".dcm";
				CoilMask_slice += ".dcm";

				
				writer_Coil->SetFileName( CoilMask_slice );
				caster->SetInput( T2gaussianFilter->GetOutput() );
				writer_Coil->SetInput( caster->GetOutput() );

				writer_T2sub->SetFileName( T2_slice );
				casterT2sub->SetInput( T2_ExtIm_filter->GetOutput() );
				writer_T2sub->SetInput( casterT2sub->GetOutput() );

				try
				{
					writer_Coil->Update();
					if (output_mode == 0)
					{
						std::cout << std::endl <<"          → Image saved as: "
							<< T2_slice << std::endl;
					}

					writer_T2sub->Update();
					if (output_mode == 0)
					{
						std::cout << std::endl <<"          → Image saved as: "
							<< CoilMask_slice << std::endl;
					}
				}
				catch( itk::ExceptionObject & err )
				{
					std::cerr << "ExceptionObject caught !" << std::endl;
					std::cerr << err << std::endl;
					return EXIT_FAILURE;
				}

			}
			catch( itk::ExceptionObject & err )
			{
				std::cerr << "ExceptionObject caught !" << std::endl;
				std::cerr << err << std::endl;
				return EXIT_FAILURE;
			}


			if (output_mode == 1)
			{
				std::cout << "\r" << "  |" << std::flush;
				for (int  c = 0; c <= i; c++)
				{
					std::cout << "==>|" << std::flush;

				}
				for (int  c = 0; c < size_z -1 - i; c++)
				{
					std::cout << "   |" << std::flush;

				}
				std::cout << "  (" 
					<< (int)((float)i/(float)(size_z-1)*100) << " %)           " << std::flush;
			}
			

		}   // else close


	} // for cycle close

	std::cout << "            Done!"<< std::endl;


	/*=========================================================================
	 
	 Segmentation Refining
	 
	 =========================================================================*/
	// Here I refine the prostete mask previously obtained.
	// To do this, I follow 2 steps:
	// 1) Subctract the periferical coil tissue mask obtained above;
	// 2) Perform Canny Levelset Segmentation to further refine the mask:
	// . 2a) 2D Canny Level Set Segmentation on T2;
	// . 2b) 3D Canny Level Set segmentaion on 2D Cannt Level Set Segmented Slices.
	//		 This generates a more smoothed and irregular volume.


	// First, I load the slices of prostate mask, the slices of coil mask and
	// the slices of the T2 (subsampled, if necessary) previously saved.

	typedef itk::ImageFileReader< OutputImageType > OutputReaderType;
	OutputReaderType::Pointer Reader_Prostate = OutputReaderType::New();
	OutputReaderType::Pointer Reader_Coil = OutputReaderType::New();
	OutputReaderType::Pointer Reader_T2sub = OutputReaderType::New();

	std::cout << std::endl << std::endl <<"→ Now Segmentation Refining is going to be performed ...  [13/19]" << std::endl;

	// ----------- FOR CYCLE START ----------- //
	for (int i = 0; i <= size_z - 1; i++)

	{	  // for cycle open

		clock.Start();					// Time probe
		sliceNumberShowed = i + 1;

		if (output_mode == 0)
		{
			std::cout << std::endl <<"     → Now processing slice "<< sliceNumberShowed << std::endl;
			std::cout << "       Loading slices ... "<< std::endl;
		}

		// Generating slice path and names to be loaded
		std::string sn;
		std::stringstream out_slice;
		out_slice << sliceNumberShowed;
		sn = out_slice.str();		// actual slice number passed to "sn"
		// Adding other parts of file name, so I have: tmp_dir_slice = tmp_dir + "/slice" + s + ".nii"
		std::string Coil_slice;
		std::string Prostate_slice;
		std::string MaskedProstate_slice;
		std::string Canny_slice;
		std::string T2_slice;

		Coil_slice += postprocessing_dir;
		Prostate_slice += postprocessing_dir;
		MaskedProstate_slice += postprocessing_dir;
		Canny_slice += segmented_dir;
		T2_slice += postprocessing_dir;

		Coil_slice += "/CoilMask_slice";
		Prostate_slice += "/slice_pp";
		MaskedProstate_slice += "/MaskedProstate_slice";
		Canny_slice += "/CannyLevelset_slice";
		T2_slice += "/T2sub_slice";

		if ( sliceNumberShowed < 10 )
		{
			Coil_slice += "0";
			Prostate_slice += "0";
			MaskedProstate_slice += "0";
			Canny_slice += "0";
			T2_slice += "0";
		}

		Coil_slice += sn;
		Prostate_slice += sn;
		MaskedProstate_slice += sn;
		Canny_slice += sn;
		T2_slice += sn;

		Coil_slice += ".dcm";
		Prostate_slice += ".dcm";
		MaskedProstate_slice += ".dcm";
		Canny_slice += ".dcm";
		T2_slice += ".dcm";

		Reader_Prostate->SetFileName( Prostate_slice );
		// Pointer to the prostatic mask (Chan & Vese + post-processing)
		OutputImageType::Pointer prostateSlice = Reader_Prostate->GetOutput();
		
		Reader_Coil->SetFileName( Coil_slice );
		// Pointer to periferical coil tissue mask
		OutputImageType::Pointer coilSlice = Reader_Coil->GetOutput();

		Reader_T2sub->SetFileName( T2_slice );
		// Pointer to T2 slices (maybe already subsampled)
		OutputImageType::Pointer T2subSlice = Reader_T2sub->GetOutput();



		// --- -- - Subtracting Periferical Coil Tissue - -- --- //
		
		// Here I use a negated mask to applay the periferical coil tissue mask
		// on the prostatic mask, in order to subtract the first from the second
		// one.


		if (output_mode == 0)
		{
			std::cout << std::endl <<"          → Subtraction of Periferical Coil Tissue is going to be performed ..."<<  std::endl;
		}

		typedef itk::MaskNegatedImageFilter< OutputImageType, OutputImageType > MaskNegatedFilterType;
		MaskNegatedFilterType::Pointer maskNegatedFilter = MaskNegatedFilterType::New();

		maskNegatedFilter->SetInput(prostateSlice);
		maskNegatedFilter->SetMaskImage(coilSlice);

		if (output_mode == 0)
		{
			std::cout << "            Done!"<< std::endl;
		}



		// --- -- - 2D Canny Levelset Segmentation - -- --- //

		// here I perform the 2D Canny levelset Segmentation to refine the mask

		if (output_mode == 0)
		{
			std::cout << std::endl <<"          → 2D Canny Levelset Segmentation is going to be performed ..."<<  std::endl;
		}

		//  The input image will be processed with a few iterations of
		//  feature-preserving diffusion.  I create a filter and set the
		//  appropriate parameters.

		typedef itk::GradientAnisotropicDiffusionImageFilter< OutputImageType,
		OutputImageType> DiffusionFilterType;
		DiffusionFilterType::Pointer gadiffusion = DiffusionFilterType::New();
		gadiffusion->SetNumberOfIterations(5);
		gadiffusion->SetTimeStep(0.125);
		gadiffusion->SetConductanceParameter(1.0);

		//  The following lines define and instantiate a
		//  CannySegmentationLevelSetImageFilter.

		typedef  itk::CannySegmentationLevelSetImageFilter< OutputImageType,
		OutputImageType > CannySegmentationLevelSetImageFilterType;
		CannySegmentationLevelSetImageFilterType::Pointer cannySegmentation =
			CannySegmentationLevelSetImageFilterType::New();
		// A propagation term
		cannySegmentation->SetAdvectionScaling( 150 );
		// To control the rigidity of the curves
		cannySegmentation->SetCurvatureScaling( 500 );
		cannySegmentation->SetPropagationScaling( 0.0 );
		// Convergence parameter (stop condition)
		// I use a slice dependent variable paramenters, to adjust the default
		// value. The aim is to lower the convergence value as I move to the 
		// external slices, where the Chan and Vese levelset is less precise.
		float default_2DCanny_RMS = 0.001;
		float var_2DCanny_RMS = abs((( i - central_slice - 1) / 2) + 1);
		cannySegmentation->SetMaximumRMSError( default_2DCanny_RMS /
		                                      var_2DCanny_RMS );
		float default_2DCanny_iters = 1500;
		float var_2DCanny_iters = abs((( i - central_slice - 1) / 6) + 1);
		cannySegmentation->SetNumberOfIterations( default_2DCanny_iters * 
		                                         var_2DCanny_iters  );

		//  There are two important parameters in the
		//  CannySegmentationLevelSetImageFilter to control the behavior of the
		//  Canny edge detection.  The \emph{variance} parameter controls the
		//  amount of Gaussian smoothing on the input image.  The \emph{threshold}
		//  parameter indicates the lowest allowed value in the output image.
		//  Thresholding is used to suppress Canny edges whose gradient magnitudes
		//  fall below a certain value.

		cannySegmentation->SetThreshold( 0 );
		// In general, higher the variance of the gaussian filter get a 
		// better segmentation, with more diffusion to proper T2 contours,
		// but also get an highr numbers of segemntation errors and
		// very irregular contours.
		// A way to take control of this is to higher the curvature scaling term.
		cannySegmentation->SetVariance( 5 );

		// Finally, it is very important to specify the isovalue of the surface in
		// the initial model input image. In a binary image, for example, the
		// isosurface is found midway between the foreground and background values.

		cannySegmentation->SetIsoSurfaceValue( 0.5 );

		gadiffusion->SetInput( T2subSlice );
		cannySegmentation->SetInput( maskNegatedFilter->GetOutput() );
		cannySegmentation->SetFeatureImage( gadiffusion->GetOutput() );


		typedef itk::BinaryThresholdImageFilter<
			OutputImageType,
		OutputImageType    >       CannyThresholdingFilterType;
		CannyThresholdingFilterType::Pointer Canny_thresholder = CannyThresholdingFilterType::New();
		Canny_thresholder->SetUpperThreshold( 10.0 );
		Canny_thresholder->SetLowerThreshold( 0.0 );
		Canny_thresholder->SetOutsideValue(  0  );
		Canny_thresholder->SetInsideValue(  255 );

		Canny_thresholder->SetInput( cannySegmentation->GetOutput() );


		try
		{
			typedef short ShortPixelType;
			typedef itk::Image< ShortPixelType, 2 > ShortImageType;
			typedef itk::CastImageFilter<
				OutputImageType,
			ShortImageType > CastFilterType;
			typedef itk::ImageFileWriter< ShortImageType > WriterType;
			WriterType::Pointer Writer_Refined = WriterType::New();
			WriterType::Pointer Writer_Canny = WriterType::New();

			CastFilterType::Pointer caster_Refined = CastFilterType::New();
			CastFilterType::Pointer caster_Canny = CastFilterType::New();


			Writer_Refined->SetFileName(MaskedProstate_slice);
			Writer_Canny->SetFileName(Canny_slice);

			caster_Refined->SetInput( maskNegatedFilter->GetOutput() );
			caster_Canny->SetInput( Canny_thresholder->GetOutput() );

			Writer_Refined->SetInput( caster_Refined->GetOutput() );
			Writer_Canny->SetInput( caster_Canny->GetOutput() );

			try
			{
				Writer_Refined->Update();
				Writer_Canny->Update();

				if (output_mode == 0)
				{
					std::cout << "            No. elpased iterations: " 
						<< cannySegmentation->GetElapsedIterations() << std::endl;
					std::cout << "            RMS change: " 
						<< cannySegmentation->GetRMSChange() << std::endl;

					std::cout << std::endl
						<<"            New Mask image without Coil tissue saved as: "
						<< MaskedProstate_slice << std::endl;
					std::cout <<"            Canny Levelset Refined Mask image saved as: "
						<< Canny_slice << std::endl;

					std::cout << std::endl << "            Done!"<< std::endl;
				}
			}
			catch( itk::ExceptionObject & err )
			{
				std::cerr << "ExceptionObject caught !" << std::endl;
				std::cerr << err << std::endl;
				return EXIT_FAILURE;
			}

		}
		catch( itk::ExceptionObject & err )
		{
			std::cerr << "ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;
			return EXIT_FAILURE;
		}



		/*======================================================================

		 Clock Probe part
		  
		 =====================================================================*/
		// Here I estimate the time needed to complete process all the slices

		int time;				// Defining the time variable 
		int min;				// Defining the minutes variable 
		int sec;				// Defining the seconds variable 

		// Calcoating the remaining time to process all slices
		// Stopping the clock probe function to get the time in this moment
		clock.Stop();

		// I calculate the mean (of all already completed cycle) at every cycle,
		// to get a more precise estimation
		// clock.GetMean gives the mean of all cycles
		time = ( clock.GetMean() * ( (size_z - i) / sample_interval) );
		if ( time > 60)
		{
			min = time / 60;
			sec = time - (min*60);
			if (output_mode == 0)
			{
				std::cout << std::endl << "     ~ Estimated time to process all remaining slices: " << min << " min " << sec << " sec" << std::endl << std::endl;
			}
			else
			{
				std::cout << "\r" << "  |" << std::flush;
				for (int  c = 0; c <= i; c++)
				{
					std::cout << "==>|" << std::flush;

				}
				for (int  c = 0; c < size_z - 1 - i; c++)
				{
					std::cout << "   |" << std::flush;

				}
				std::cout << "  (" 
					<< min << " min " << sec << " sec left)           " << std::flush;
			}
		}	
		else
		{
			sec = time;
			if (output_mode == 0)
			{
				std::cout << std::endl << "     ~ Estimated time to process all remaining slices: " << sec << " sec" << std::endl << std::endl;
			}
			else
			{
				std::cout << "\r" << "  |" << std::flush;
				for (int  c = 0; c <= i; c++)
				{
					std::cout << "==>|" << std::flush;

				}
				for (int  c = 0; c < size_z - 1 - i; c++)
				{
					std::cout << "   |" << std::flush;

				}
				std::cout << "  (" 
					<< sec << " sec left)           " << std::flush;
			}
		}



	}		// for cycle close
	// ----------- FOR CYCLE END ----------- //

	std::cout << "            Done!"<< std::endl;
	

	/*=========================================================================
	 
	 dcm2nii_dwi - Canny volume
	 
	 =========================================================================*/

	// Now I proceed to rebuild the volume from the Canny Levelset slices 
	//previously generated.
	// Since the dcm writing cause a lost of some dicom information, 
	//I use the slices' file names generated to rebuld the volume. 

	if (output_mode == 0)
	{
		std::cout << std::endl << std::endl <<"→ Now Canny Levelset Volume build is going to be performed ...  [14/19]" << std::endl;
	}

	d2n_reader->SetImageIO( dicomIO );

	nameGenerator->SetUseSeriesDetails( false );
	nameGenerator->SetDirectory( segmented_dir );
	//std::string prostate_volume;

	try
	{
		if (output_mode == 0)
		{
			std::cout << std::endl << "          Analizing directory..." << std::endl;
			std::cout << "          Found following DICOM Series: ";
			std::cout << std::endl;
		}

		typedef std::vector< std::string >    SeriesIdContainer;
		typedef std::vector< std::string >   FileNamesContainer;
		FileNamesContainer fileNames;

		const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

		SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
		SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
		while( seriesItr != seriesEnd )
		{
			if (output_mode == 0)
			{
				std::cout << "          " << seriesItr->c_str() << std::endl;
			}
			fileNames.insert(fileNames.end(), nameGenerator->GetFileNames(seriesItr->c_str()).begin(), nameGenerator->GetFileNames(seriesItr->c_str()).end());
			seriesItr++;
		}

		std::sort(fileNames.begin(), fileNames.end(), StringLessThen);

		d2n_reader->SetFileNames( fileNames );

		try
		{
			d2n_reader->Update();
		}
		catch (itk::ExceptionObject &ex)
		{
			std::cout << ex << std::endl;
			return EXIT_FAILURE;
		}

		typedef itk::OrientImageFilter<ImageType,ImageType> OrienterType;
		OrienterType::Pointer orienter = OrienterType::New();

		orienter->UseImageDirectionOn();
		orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
		orienter->SetInput(d2n_reader->GetOutput());

		typedef itk::ImageFileWriter< ImageType > WriterType;
		WriterType::Pointer writer = WriterType::New();

		// Generating volume output file name, usign the output directory given
		segmented_vol = segmented_dir;		// passing output directory to "tmp_dir_volume"
		segmented_vol += "/CannyLevelset_Volume.nii";		// post-adding "/---.nii"

		writer->SetFileName( segmented_vol );

		writer->SetInput( orienter->GetOutput() );

		if (output_mode == 0)
		{
			std::cout << std::endl << "          Generating the new volume ... " << std::endl;
			std::cout << "          New volume saved as: " << std::endl;
			std::cout << "          " << segmented_vol << std::endl;
			std::cout << "          Done!" << std::endl;
		}

		try
		{
			writer->Update();
		}
		catch (itk::ExceptionObject &ex)
		{
			std::cout << ex << std::endl;
			return EXIT_FAILURE;
		}
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}
	

	
	/*=========================================================================
	 
	 3D Canny Levelset Segmantation
	 
	 =========================================================================*/
	// Now, to further refine the 2D segmentation, I proceed to applay a 3D
	// segmentation of the volume with the Canny Levelset.
	// In this way I obtaina more smooth and uniform volume, because now is
	// also processed the z direction.

	std::cout << std::endl <<"→ Now 3D Canny Levelset Segmentation Refining is going to be performed ...  [15/19]" << std::endl;

	// Loading the volume previously generated
	T2_ReaderType::Pointer Reader_Canny3D = T2_ReaderType::New();
	Reader_Canny3D->SetFileName(segmented_vol);
	Reader_Canny3D->Update();


	//  The input image will be processed with a few iterations of
	//  feature-preserving diffusion.  I create a filter and set the
	//  appropriate parameters.

	typedef itk::GradientAnisotropicDiffusionImageFilter< InputImageType,
	InputImageType> DiffusionFilterType3D;
	DiffusionFilterType3D::Pointer gadiffusion3D = DiffusionFilterType3D::New();
	gadiffusion3D->SetNumberOfIterations(5);
	gadiffusion3D->SetTimeStep(0.05);
	gadiffusion3D->SetConductanceParameter(1.0);

	//  The following lines define and instantiate a
	//  CannySegmentationLevelSetImageFilter.

	typedef  itk::CannySegmentationLevelSetImageFilter< InputImageType,
	InputImageType > CannySegmentationLevelSetImageFilterType3D;
	CannySegmentationLevelSetImageFilterType3D::Pointer cannySegmentation3D =
		CannySegmentationLevelSetImageFilterType3D::New();

	cannySegmentation3D->SetAdvectionScaling( 150 );
	cannySegmentation3D->SetCurvatureScaling( 500 );
	cannySegmentation3D->SetPropagationScaling( 0.0 );

	cannySegmentation3D->SetMaximumRMSError( 0.02 );
	cannySegmentation3D->SetNumberOfIterations( 1000 );

	//  There are two important parameters in the
	//  CannySegmentationLevelSetImageFilter to control the behavior of the
	//  Canny edge detection.  The \emph{variance} parameter controls the
	//  amount of Gaussian smoothing on the input image.  The \emph{threshold}
	//  parameter indicates the lowest allowed value in the output image.
	//  Thresholding is used to suppress Canny edges whose gradient magnitudes
	//  fall below a certain value.

	cannySegmentation3D->SetThreshold( 0 );
	cannySegmentation3D->SetVariance( 20 );

	// Finally, it is very important to specify the isovalue of the surface in
	// the initial model input image. In a binary image, for example, the
	// isosurface is found midway between the foreground and background values.

	cannySegmentation3D->SetIsoSurfaceValue( 0.5 );



	// --- -- - Change Image Information - -- --- //
	
	// Now a lot of image information control is needed due to different image informations
	// (I previously have taken care of it but was a 2D case, so z direction in not
	// fixed) and due to a lot of bugs in ITK.


	// Here I ensure that the T2 volume and the Canny volume have the same directions

	typedef itk::ChangeInformationImageFilter< InputImageType > ChDirection_FilterType;
	ChDirection_FilterType::Pointer ChDirection_filter = ChDirection_FilterType::New();

	//InputImageType::Pointer T2Volume_DirectionImage = T2_featureImage;
	InputImageType::DirectionType T2Volume_direction = T2_featureImage->GetDirection();
	if (output_mode == 0)
	{
		std::cout << std::endl << "     T2Volume_ direction is: "<< std::endl;
		std::cout << T2Volume_direction << std::endl;
	}

	//InputImageType::Pointer CannyVolume_DirectionImage = Reader_Canny3D->GetOutput();
	InputImageType::DirectionType CannyVolume_direction = Reader_Canny3D->GetOutput()->GetDirection();
	if (output_mode == 0)
	{
		std::cout << std::endl << "     CannyVolume_ direction is: "<< std::endl;
		std::cout << CannyVolume_direction << std::endl;
	}

	T2Volume_direction = CannyVolume_direction;

	if (output_mode == 0)
	{
		std::cout << "     → Fixing direction" << std::endl;
	}
	ChDirection_filter->SetOutputDirection( T2Volume_direction );
	ChDirection_filter->ChangeDirectionOn();

	ChDirection_filter->SetInput( T2_featureImage);
	ChDirection_filter->Update();
	if (output_mode == 0)
	{
		std::cout << "          New correct  direction is: "<< std::endl;
		std::cout<< T2Volume_direction << std::endl;
		std::cout << "          Fixed!"<< std::endl<< std::endl;
	}
	
	gadiffusion3D->SetInput( ChDirection_filter->GetOutput() );

	
	// Here I proceed to control spacing and origin of the T2 volume

	// Now some control precesses are needed:
	// Origin must be [0,0,0] (to avoid "requested region out of laregest possible region" issue)
	// T2 Spacing (aka PixelResolutin) must be equal to the b0 MRI.


	InputImageType::Pointer Canny3D_inputImage = Reader_Canny3D->GetOutput();

	InputImageType::PointType Canny3D_iorigin = Canny3D_inputImage->GetOrigin();
	if (output_mode == 0)
	{
		std::cout << "     → Image origin control now is taking place..." << std::endl;
		std::cout << "       input image origin: " << iorigin << std::endl;
	}

	typedef itk::ChangeInformationImageFilter< InputImageType > ChInfo_FilterType;
	ChInfo_FilterType::Pointer Canny3D_ChInfo_filter = ChInfo_FilterType::New();
	InputImageType::SpacingType Canny3D_ext_spacing = Canny3D_inputImage->GetSpacing();

	// Fixing origin to (0,0,0) to avoid "out of largest possible region" error.
	InputImageType::PointType Canny3D_origin = 0;
	if (output_mode == 0)
	{
		std::cout << "       applying new origin: " << Canny3D_origin << std::endl;
	}
	Canny3D_ChInfo_filter->SetOutputOrigin( Canny3D_origin );	
	Canny3D_ChInfo_filter->ChangeOriginOn();
	Canny3D_ChInfo_filter->SetInput( Canny3D_inputImage );

	InputImageType::Pointer Canny3D_featureImage = Canny3D_ChInfo_filter->GetOutput();
	InputImageType::PointType Canny3D_forigin = Canny3D_featureImage->GetOrigin();
	if (output_mode == 0)
	{
		std::cout << "       output image origin: " << Canny3D_forigin << std::endl;
		std::cout << "       Done!" << std::endl;
		std::cout << " " << std::endl;
	}

	Canny3D_ChInfo_filter->UpdateLargestPossibleRegion();
	// --- -- -                  - -- --- //

	std::cout << "     → Running Levelset (this may take a while) ..." << std::flush;

	cannySegmentation3D->SetInput( Canny3D_featureImage );
	cannySegmentation3D->SetFeatureImage( gadiffusion3D->GetOutput() );


	typedef itk::BinaryThresholdImageFilter<
		InputImageType,
	InputImageType    >       CannyThresholdingFilterType3D;
	CannyThresholdingFilterType3D::Pointer Canny_thresholder3D = CannyThresholdingFilterType3D::New();
	Canny_thresholder3D->SetUpperThreshold( 10.0 );
	Canny_thresholder3D->SetLowerThreshold( 0.0 );
	Canny_thresholder3D->SetOutsideValue(  0  );
	Canny_thresholder3D->SetInsideValue(  255 );

	Canny_thresholder3D->SetInput( cannySegmentation3D->GetOutput() );

	try
	{
		typedef itk::ImageFileWriter< InputImageType > WriterType;
		WriterType::Pointer Writer_Canny3D = WriterType::New();

		Writer_Canny3D->SetFileName(segmented_vol);

		Writer_Canny3D->SetInput( Canny_thresholder3D->GetOutput() );

		try
		{
			Writer_Canny3D->Update();
			if (output_mode == 0)
			{
				std::cout << std::endl << "          No. elpased iterations: " << cannySegmentation3D->GetElapsedIterations() << std::endl;
				std::cout << "          RMS change: " << cannySegmentation3D->GetRMSChange() << std::endl;

				std::cout << std::endl <<"          →  3D Canny Levelset Image saved as: "<< segmented_vol << std::endl;
			}
			std::cout << "          Done!" << std::endl;
		}
		catch( itk::ExceptionObject & err )
		{
			std::cerr << "ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;
			return EXIT_FAILURE;
		}

	}
	catch( itk::ExceptionObject & err )
	{
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
		return EXIT_FAILURE;
	}



	/*=========================================================================

    Post_processing

    =========================================================================*/

	// Here I perform some post-processing operations on the Canny volume.
	// First, I have again to extrat each slice.
	// Then I perform:
	// · Connected Components
	// · Hole Filling

	std::cout << std::endl <<"→ Now Post-processing operations on 3D Canny Levelset Segmentation is going to be performed ...  [16/19]" << std::endl;
	
	// --- -- - Slice Extraction - -- --- //

	// here I instantiate some variables for a successive slice extraction

	FilterType::Pointer Canny3D_ExtIm_filter = FilterType::New();
	Canny3D_ExtIm_filter->InPlaceOff();
	Canny3D_ExtIm_filter->SetDirectionCollapseToSubmatrix();
	typedef itk::ImageFileReader< InternalImageType > Canny3D_ReaderType;
	Canny3D_ReaderType::Pointer Canny3D_ExtIm_reader = Canny3D_ReaderType::New();
	Canny3D_ExtIm_reader->SetFileName(segmented_vol);
	Canny3D_ExtIm_reader->Update();

	// Now I need to perform 2D filtering, so T2 volume slice extraction is 
	// needed.

	InputImageType::RegionType Canny3D_inputRegion = Canny3D_ExtIm_reader->GetOutput()->GetLargestPossibleRegion();
	InputImageType::SizeType Canny3D_size = T2_inputRegion.GetSize();
	// Collapsing z direction
	Canny3D_size[2]=0;
	InputImageType::IndexType Canny3D_start = T2_inputRegion.GetIndex();

	// ----------- FOR CYCLE START ----------- //
	for ( int i = 0; i <= size_z - 1; i++  )

	{		// for cycle open

		// Starting slice extraction
		Canny3D_start[2] = i;

		// Using slicenNumberShowed instead of sliceNumber produces a more human
		// readeable output and gives coerency with slices generated file names
		// with other IRCCS study (ad-hoc fix)
		sliceNumberShowed = i + 1;
		if (output_mode == 0)
		{
			std::cout <<"     → Now processing slice "<< sliceNumberShowed << std::endl;
		}

		// Finally, an \doxygen{ImageRegion} object is created and initialized with
		// the start and size I just prepared using the slice information.

		InputImageType::RegionType Canny3D_desiredRegion;
		Canny3D_desiredRegion.SetSize( Canny3D_size );
		Canny3D_desiredRegion.SetIndex( Canny3D_start );

		// Then the region is passed to the filter using the
		// SetExtractionRegion() method.

		Canny3D_ExtIm_filter->SetExtractionRegion( Canny3D_desiredRegion );

		// I extract the segmente slices from the levelset
		Canny3D_ExtIm_filter->SetInput( Canny3D_ExtIm_reader->GetOutput() );
		Canny3D_ExtIm_filter->UpdateLargestPossibleRegion();

		// Generating slice path and names to be loaded

		std::string sn;
		std::stringstream out_slice;
		out_slice << sliceNumberShowed;
		sn = out_slice.str();		// actual slice number passed to "sn"
		// Adding other parts of file name, so I have: tmp_dir_slice = tmp_dir + "/slice" + s + ".nii"
		std::string Coil_slice;
		std::string Canny_slice;

		Coil_slice += postprocessing_dir;
		Canny_slice += segmented_dir;

		Coil_slice += "/CoilMask_slice";
		Canny_slice += "/CannyLevelset_slice";

		if ( sliceNumberShowed < 10 )
		{
			Coil_slice += "0";
			Canny_slice += "0";

		}

		Coil_slice += sn;
		Canny_slice += sn;

		Coil_slice += ".dcm";
		Canny_slice += ".dcm";

		Reader_Coil->SetFileName( Coil_slice );
		// Pointer to periferical coil tissue mask
		OutputImageType::Pointer coilSlice = Reader_Coil->GetOutput();

		

		// --- -- - Connected Components - -- --- //
		
		// Here I proceed to mantain only the largest mask in case I have some
		// scattered and sparse pieces of mask

		// Short pixel type is required by ConnectedComponentsFilter because
		// an integral pixel type is needed.
		typedef  short shortPixelType;
		typedef itk::Image< shortPixelType, 2 >  shortImageType;
		
		typedef itk::ConnectedComponentImageFilter <OutputImageType, shortImageType >
			ConnectedComponentImageFilterType;
		ConnectedComponentImageFilterType::Pointer connected = 
			ConnectedComponentImageFilterType::New ();
		connected->SetInput(Canny3D_ExtIm_filter->GetOutput());
		connected->Update();
		if (output_mode == 0)
		{
			std::cout << "          → Connected Components filter operations is going to be performed ..." <<  std::endl;
			std::cout << "            Number of objects: " << connected->GetObjectCount() << std::endl;
		}
		typedef itk::LabelShapeKeepNObjectsImageFilter< shortImageType 
			> LabelShapeKeepNObjectsImageFilterType;
		LabelShapeKeepNObjectsImageFilterType::Pointer 
			labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
		labelShapeKeepNObjectsImageFilter->SetInput( connected->GetOutput() );
		labelShapeKeepNObjectsImageFilter->SetBackgroundValue( 0 );
		// To mantain only the (one) main mask
		labelShapeKeepNObjectsImageFilter->SetNumberOfObjects( 1 );
		// This is the critera to wich mask mantain. This is the maximum number
		// of pixels.
		labelShapeKeepNObjectsImageFilter->
			SetAttribute( LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
		typedef itk::RescaleIntensityImageFilter< shortImageType, OutputImageType
			> RescaleFilterType;
		RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
		rescaleFilter->SetOutputMinimum(0);
		rescaleFilter->SetOutputMaximum(1);
		rescaleFilter->SetInput(labelShapeKeepNObjectsImageFilter->GetOutput());
		if (output_mode == 0)
		{
			std::cout << "            Done!" <<  std::endl;
		}



		// --- -- - Hole Filling - -- --- //
		
		// Here I fill holes in the mask

		if (output_mode == 0)
		{
			std::cout << "          → Hole Filling filter operations is going to be performed ..." <<  std::endl;
		}
		typedef itk::BinaryFillholeImageFilter <OutputImageType>
			BinaryFillholeImageFilterType;
		BinaryFillholeImageFilterType::Pointer fillhole = 
			BinaryFillholeImageFilterType::New ();
		fillhole->SetInput(rescaleFilter->GetOutput());
		fillhole->SetForegroundValue(1);
		fillhole->Update();
		if (output_mode == 0)
		{		
			std::cout << "            Done!" <<  std::endl;
		}

		try
		{
			typedef short ShortPixelType;
			typedef itk::Image< ShortPixelType, 2 > ShortImageType;
			typedef itk::CastImageFilter<
				OutputImageType,
			ShortImageType > CastFilterType;
			typedef itk::ImageFileWriter< ShortImageType > WriterType;
			WriterType::Pointer Writer_Canny_pp = WriterType::New();

			CastFilterType::Pointer caster_Canny_pp = CastFilterType::New();
			Writer_Canny_pp->SetFileName(Canny_slice);
			caster_Canny_pp->SetInput( fillhole->GetOutput() );
			Writer_Canny_pp->SetInput( caster_Canny_pp->GetOutput() );

			try
			{

				Writer_Canny_pp->Update();
				if (output_mode == 0)
				{
					std::cout <<"            Post-processing Canny Levelset Refined Mask image saved as: "
						<< Canny_slice << std::endl;

					std::cout << std::endl << "            Done!"<< std::endl;
				}
			}
			catch( itk::ExceptionObject & err )
			{
				std::cerr << "ExceptionObject caught !" << std::endl;
				std::cerr << err << std::endl;
				return EXIT_FAILURE;
			}

		}
		catch( itk::ExceptionObject & err )
		{
			std::cerr << "ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;
			return EXIT_FAILURE;
		}

		if (output_mode == 1)
		{
			std::cout << "\r" << "  |" << std::flush;
			for (int  c = 0; c <= i; c++)
			{
				std::cout << "==>|" << std::flush;

			}
			for (int  c = 0; c < size_z -1 - i; c++)
			{
				std::cout << "   |" << std::flush;

			}
			std::cout << "  (" 
				<< (int)((float)i/(float)(size_z-1)*100) << " %)           " << std::flush;
		}
		
	}

	std::cout << "            Done!"<< std::endl;




	/*=========================================================================
	  
	 Blank Black Image Creation part
	  
	 =========================================================================*/
	// Here I create a blank black image, to replace the exculded slices in
	// the new volume being create.

	// DICOM type only supports short pixel type.
	// Prevyously I had used float type becouse same filters only support float
	// pixel type.
	typedef itk::Image< short, 2 >  dcmImageType;

	dcmImageType::RegionType black_region;
	dcmImageType::IndexType black_start;
	black_start[0] = 0;
	black_start[1] = 0;

	dcmImageType::SizeType black_size;
	black_size[0] = size[0];
	black_size[1] = size[1];

	black_region.SetSize(black_size);
	black_region.SetIndex(black_start);

	dcmImageType::Pointer black_image = dcmImageType::New();
	black_image->SetRegions(black_region);
	black_image->Allocate();
	black_image->FillBuffer(0);
	


	/*=========================================================================
	  
	 New Slices Creation part
	  
	 =========================================================================*/
	// Here I create the new slices one by one.
	// Excluded slices is going to be all black.
	// Included slices is going to be equal to the original.
	// So I re-create a set of slice01, .... , slice24
	// and saved to local disc, ready to be loaded in the next step,
	// to build the new volume wich contains only the prostate.

	if (output_mode == 0)
	{
		std::cout << std::endl << "→ Now Creation of the new volume containing only the prostate is going to be performed ...  [17/19]" << std::endl;
	}

	// Here I create a new "New_Volume" directory, inside the core directory,
	// in which the new slices is goping to be saved.
	std::string new_volume_dir;
	// Passing output directory to "segmented_dir"
	new_volume_dir = core_dir.str();
	// Adding the new folder
	new_volume_dir += "/Final_Segmented_Volume";
	// Converting string to const char, required by "CreateDirectory"
	const char* char_new_volume_dir = new_volume_dir.c_str();
	// Creating the new "Segmented" directory
	itk::FileTools::CreateDirectory (char_new_volume_dir);

	if (output_mode == 0)
	{
		std::cout << "          Created the directory: " << char_new_volume_dir << std::endl << std::endl;
		std::cout << "          Generating new slices ... " << std::endl;
	}

	// Here I set a for cycle to process all slices
	// k goes from 1 to 24 (not from 0 to 23) 'cause I need the sliceNumberShowed
	// to read the original slices and to generate the new slices.
	for(int k = 1; k <= size_z; k++)
	{

		// Generating actual slice numer string
		std::string sn;
		std::stringstream out_slice;
		out_slice << k;
		sn = out_slice.str();		// actual slice number passed to "sn"

		// Adding other parts of file name, so I have: tmp_dir_slice = new_volume_dir + "/slice" + sn + ".dcm"
		std::string tmp_dir_slice;
		tmp_dir_slice += new_volume_dir;
		tmp_dir_slice += "/slice";
		if ( k < 10 )
		{
			// I want a name generation like this:
			// slice01, slice02, ... , slice10, slice11, ...
			tmp_dir_slice += "0";
		}
		tmp_dir_slice += sn;
		// Output slice format. dcm requires a casting operation, 
		// becouse float is not supported.
		tmp_dir_slice += ".dcm";

		if (output_mode == 0)
		{
			std::cout << "          Writing slice as:   " << tmp_dir_slice << std::endl;
		}

		// Instantiating reader and writer for the new slices.
		typedef itk::ImageFileWriter< dcmImageType > newSlice_WriterType;
		newSlice_WriterType::Pointer newSlice_writer = newSlice_WriterType::New();
		typedef itk::ImageFileReader< dcmImageType > newSlice_ReaderType;
		newSlice_ReaderType::Pointer newSlice_reader = newSlice_ReaderType::New();

		// Setting the file name generated before
		newSlice_writer->SetFileName( tmp_dir_slice );

		// Defining when the original slices must be used
		if (k >= lower_slice_include_value && k<= upper_slice_include_value)
		{

			// Generating the file names to be read, so I kane keep the cropped slices
			// from the cropped folder.
			std::string b0_dir_slice;
			b0_dir_slice += segmented_dir;
			b0_dir_slice += "/CannyLevelset_slice";
			if ( k < 10 )
			{
				b0_dir_slice += "0";		// I want a name generation like this: slice01, slice02, ... , slice10, slice11, ...
			}
			b0_dir_slice += sn;
			b0_dir_slice += ".dcm";		// Output slice format. dcm requires a casting operation, becouse float is not supported.

			// I read the original slice, one by one
			newSlice_reader->SetFileName( b0_dir_slice );
			newSlice_reader->Update();
			// And now I pass it to the writer
			newSlice_writer->SetInput( newSlice_reader->GetOutput() );

		}

		else
		{

			// If I are not in the included slice case,
			// I pass the black image before generated, to the writer.
			newSlice_writer->SetInput( black_image );

		}

		// Finally, I can invoke the update on the new slice writer
		newSlice_writer->Update();

	}				// for cycle close
	

	


	/*=========================================================================
	 
	 dcm2nii_dwi - post-processed Canny
	 
	 =========================================================================*/

	// Now I proceed to rebuild the volume from the post-processed Canny
	// Levelset slices previously generated.
	// Since the dcm writing cause a lost of some dicom information, 
	//I use the slices' file names generated to rebuld the volume. 

	if (output_mode == 0)
	{
		std::cout << std::endl << std::endl <<"→ Now Post-processed Canny Levelset Volume build is going to be performed ...  [18/19]" << std::endl;
	}

	std::string final_volume;

	d2n_reader->SetImageIO( dicomIO );

	nameGenerator->SetUseSeriesDetails( false );
	nameGenerator->SetDirectory( new_volume_dir );
	//std::string prostate_volume;

	try
	{
		if (output_mode == 0)
		{
			std::cout << std::endl << "          Analizing directory..." << std::endl;
			std::cout << "          Found following DICOM Series: ";
			std::cout << std::endl;
		}

		typedef std::vector< std::string >    SeriesIdContainer;
		typedef std::vector< std::string >   FileNamesContainer;
		FileNamesContainer fileNames;

		const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

		SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
		SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
		while( seriesItr != seriesEnd )
		{
			if (output_mode == 0)
			{
				std::cout << "          " << seriesItr->c_str() << std::endl;
			}
			fileNames.insert(fileNames.end(), nameGenerator->
			                 GetFileNames(seriesItr->c_str()).begin(),
			                 nameGenerator->GetFileNames(seriesItr->c_str()).end());
			seriesItr++;
		}

		std::sort(fileNames.begin(), fileNames.end(), StringLessThen);

		d2n_reader->SetFileNames( fileNames );

		try
		{
			d2n_reader->Update();
		}
		catch (itk::ExceptionObject &ex)
		{
			std::cout << ex << std::endl;
			return EXIT_FAILURE;
		}

		typedef itk::OrientImageFilter<ImageType,ImageType> OrienterType;
		OrienterType::Pointer orienter = OrienterType::New();

		orienter->UseImageDirectionOn();
		orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
		orienter->SetInput(d2n_reader->GetOutput());

		typedef itk::ImageFileWriter< ImageType > WriterType;
		WriterType::Pointer writer = WriterType::New();
		
		 // Generating volume output file name, usign the output directory given


		 final_volume = core_dir.str();		// passing output directory to "tmp_dir_volume"
		 final_volume += "/Final_Segmented_Volume_Prostate.nii";	// post-adding "/---.nii"
		 
		writer->SetFileName( final_volume );

		writer->SetInput( orienter->GetOutput() );

		if (output_mode == 0)
		{
			std::cout << std::endl << "          Generating the new volume ... " << std::endl;
			std::cout << "          New volume saved as: " << std::endl;
			std::cout << "          " << final_volume << std::endl;
			std::cout << "          Done!" << std::endl;
		}

		try
		{
			writer->Update();
		}
		catch (itk::ExceptionObject &ex)
		{
			std::cout << ex << std::endl;
			return EXIT_FAILURE;
		}
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}




	/*=========================================================================

	 Image Information Match

	 =========================================================================*/
	// Now I proceed to change size, spacing and all the other mask image informations
	// to match the T2 ones.
	// This is necessary to furher comapre mask with T2, and filter some image
	// informations (such as tumor) find on T2. 

	if (output_mode == 0)
	{
		std::cout << std::endl << std::endl <<"→ Now Resampling and Change Information operations to match T2 image is going to be performed ... [19/19]" << std::endl;
	}

	std::string fixed_final_volume;
	
	try
	{

		// Here I instantiate readers for T2 original volume and for
		// final segmented mask volume
		T2_ReaderType::Pointer T2OriginalVoulme_reader = T2_ReaderType::New();
		T2_ReaderType::Pointer FinalSegmentedVoulme_reader = T2_ReaderType::New();

		T2OriginalVoulme_reader->SetFileName( T2_volume );
		T2OriginalVoulme_reader->Update();

		FinalSegmentedVoulme_reader->SetFileName( final_volume );
		FinalSegmentedVoulme_reader->Update();


		// Now I instantiate the resample filter		
		typedef itk::ResampleImageFilter< InternalImageType, ImageType >  FilterType;
		FilterType::Pointer resampler = FilterType::New();

		typedef itk::NearestNeighborInterpolateImageFunction< InternalImageType, double >  InterpolatorType;
		InterpolatorType::Pointer interpolator = InterpolatorType::New();

		// Here I set the upsample (/subsample) parameters to scale the mask
		// to match T2 dimensions.
		typedef itk::ScaleTransform<double, 3> TransformType;
		TransformType::Pointer scaleTransform = TransformType::New();
		itk::FixedArray<float, 3> scale;

		// Here I get the size of mask and T2, and I find the new scal correction
		// factor. Z scale must be equal.
		InputImageType::RegionType T2OriginalVolume = T2OriginalVoulme_reader->GetOutput()->GetLargestPossibleRegion();
		InputImageType::RegionType FinalSegmentedVoulme = FinalSegmentedVoulme_reader->GetOutput()->GetLargestPossibleRegion();
		scale[0] = T2OriginalVolume.GetSize()[0] / FinalSegmentedVoulme.GetSize()[0];
		scale[1] = T2OriginalVolume.GetSize()[1] / FinalSegmentedVoulme.GetSize()[1];
		scale[2]=1;
		scaleTransform->SetScale(scale);

		// Now I adjust the spacing in function of the new size
		InternalImageType::SpacingType spacing;
		spacing[0]=FinalSegmentedVoulme_reader->GetOutput()->GetSpacing()[0]*(1/(scale[0]*scale[0]));
		spacing[1]=FinalSegmentedVoulme_reader->GetOutput()->GetSpacing()[1]*(1/(scale[0]*scale[0]));
		spacing[2]=FinalSegmentedVoulme_reader->GetOutput()->GetSpacing()[2];
		
		resampler->SetTransform(          scaleTransform                  );
		resampler->SetInterpolator(       interpolator               );
		resampler->SetDefaultPixelValue(  0                          );
		resampler->SetSize(               T2OriginalVoulme_reader->GetOutput()->GetLargestPossibleRegion().GetSize());
		resampler->SetOutputSpacing(      spacing      );
		resampler->SetInput(              FinalSegmentedVoulme_reader->GetOutput()   );
		resampler->Update();



		// --- -- - Change Image Information - -- --- //
		
		// Now I change the information image to match to all T2 informations.

		typedef itk::ChangeInformationImageFilter< InputImageType > ChDirection_FilterType;
		ChDirection_FilterType::Pointer ChDirection_filter = ChDirection_FilterType::New();

		//std::cout << std::endl << "     T2Volume_ direction is: "<< std::endl;
		//std::cout << T2OriginalVoulme_reader->GetOutput()->GetDirection() << std::endl;
		//std::cout << "     → Fixing direction" << std::endl;
		//ChDirection_filter->SetOutputDirection( T2OriginalVoulme_reader->GetOutput()->GetDirection() );
		//ChDirection_filter->ChangeDirectionOn();
		ChDirection_filter->SetOutputOrigin( T2OriginalVoulme_reader->GetOutput()->GetOrigin() );
		ChDirection_filter->ChangeOriginOn();
		ChDirection_filter->SetOutputSpacing( T2OriginalVoulme_reader->GetOutput()->GetSpacing() );
		ChDirection_filter->ChangeSpacingOn();
		ChDirection_filter->SetInput( resampler->GetOutput());
		
		ChDirection_filter->Update();
		
		//std::cout << "          New correct  direction is: "<< std::endl;
		//std::cout<< ChDirection_filter->GetOutput()->GetDirection() << std::endl;
		//std::cout << "          Fixed!"<< std::endl<< std::endl;
		if (output_mode == 0)
		{
			std::cout << "          New correct  origin is: "<< std::endl;
			std::cout<< ChDirection_filter->GetOutput()->GetOrigin() << std::endl;
			std::cout << "          Fixed!"<< std::endl<< std::endl;
			std::cout << "          New correct  spacing is: "<< std::endl;
			std::cout<< ChDirection_filter->GetOutput()->GetSpacing() << std::endl;
			std::cout << "          Fixed!"<< std::endl<< std::endl;
		}

		// Here I write the volume image of final segmented one.

		typedef itk::OrientImageFilter<InputImageType,ImageType> OrienterType;
		OrienterType::Pointer orienter = OrienterType::New();

		orienter->UseImageDirectionOn();
		orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);

		orienter->SetInput(ChDirection_filter->GetOutput());	
		orienter->Update();

		WriterType::Pointer Resampler_writer = WriterType::New();

		fixed_final_volume = core_dir.str();		// passing output directory to "tmp_dir_volume"
		fixed_final_volume += "/Eval+Esc+H_Fixed_Final_Segmented_Volume_Prostate.tiff";		// post-adding "/---.nii"
		
		Resampler_writer->SetFileName(fixed_final_volume);
		Resampler_writer->SetInput(orienter->GetOutput());
		Resampler_writer->Update();

		if (output_mode == 0)
		{
			std::cout << "  New Fixed Volume saved as: " << fixed_final_volume << std::endl;
			std::cout << std::endl << std::endl <<"  Done!" << std::endl;
		}

	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}




	/*=========================================================================
	  
	 Final Report
	  
	 =========================================================================*/
	// Here I print a final report, with warnings and informations
	std::cout << std::endl << "------ ----- ---- --- -- - Final Report: - -- --- ---- ----- ------" << std::endl;

	// --- -- - WARNINGS - -- --- //

	if (upper_limit_found == false)
	{
		std::cout << std::endl << "[!] WARNING! The Automatic Slices Esclusion Algorithm as failed!" << std::endl;
		std::cout << "          Top prostatic estension limit not found! " << std::endl;
		std::cout << "          The segmented volume will include the slices from "
			<< size_z/2 << " to " << upper_slice_include_value << std::endl << std::endl;
	}
	if (lower_limit_found == false)
	{
		std::cout << std::endl << "[!] WARNING! The Automatic Slices Esclusion Algorithm as failed!" << std::endl;
		std::cout << "          Bottom prostatic estension limit not found! " << std::endl;
		std::cout << "          The segmented volume will include the slices from "
			<< lower_slice_include_value << " to " << size_z/2 << std::endl << std::endl;
	}
	if (ThRegionGrowingFailed == true)
	{
		std::cout << std::endl << "[!] WARNING! The Automatic Region Growing Threshold Control Algorithm as failed!" << std::endl;
		std::cout << "          Some Coil Tissue has not been removed! (see full output for more informations)" << std::endl;
	}
	if (H_search_fail_central == true)
	{
		std::cout << std::endl << "[!] WARNING! The Automatic Heaviside Function Search Algorithm as failed!" << std::endl;
		std::cout << "          Central sub-volume segmentation is not optimal!" << std::endl;
	}
	if (H_search_fail_bottom == true)
	{
		std::cout << std::endl << "[!] WARNING! The Automatic Heaviside Function Search Algorithm as failed!" << std::endl;
		std::cout << "          Bottom sub-volume segmentation is not optimal!" << std::endl;
	}
	if (H_search_fail_top == true)
	{
		std::cout << std::endl << "[!] WARNING! The Automatic Heaviside Function Search Algorithm as failed!" << std::endl;
		std::cout << "          Top sub-volume segmentation is not optimal!" << std::endl;
	}	


	// --- -- - GENERAL INFORMATIONS - -- --- //

	// Here the total time is calculated and printed

	int time;				// Defining the time variable 
	int min;				// Defining the minutes variable 
	int sec;				// Defining the seconds variable 
	// Calcoating the remaining time to process all slices
	clock_all.Stop();							// Stopping the clock probe function to get the time in this moment
	time = clock_all.GetTotal();
	if ( time > 60)
	{
		min = time / 60;
		sec = time - (min*60);
	}
	else
	{
		sec = time;
	}

	std::cout << std::endl << "[i] Total time: " << min << " min " << sec << " sec " << std::endl;

	std::cout << std::endl << "[i] Final Segmented Volume: " << fixed_final_volume << std::endl;

	std::cout << std::endl << "[V] Prostate Automatic Segmentation SUCCESSFULLY completed!" << std::endl << std::endl;


	return EXIT_SUCCESS;

}

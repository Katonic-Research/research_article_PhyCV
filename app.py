import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='Interactive research article using Streamlit',  
    layout = 'centered', 
    initial_sidebar_state = 'auto'
)


st.markdown('Research article')

st.title('PhyCV: The First Physics-inspired Computer Vision Library')

st.header('Author')
st.warning('''
Yiming Zhou, Callen MacPhee, Madhuri Suthar, and Bahram Jalali Electrical and Computer Engineering Department
University of California, Los Angeles

''')

st.header('Abstract')
st.info('''
PhyCV is the first computer vision library which utilizes algorithms directly derived from the equations of physics governing physical phenomena. The algorithms appearing in the current release emulate, in a metaphoric sense, the propagation of light through a physical medium with natural and engineered diffractive properties followed by coherent detection. Unlike traditional algorithms that are a sequence of hand-crafted empirical rules, physics-inspired algorithms leverage physical laws of nature as blueprints for inventing algorithms. In addition, these algorithms have the potential to be implemented in real physical devices for fast and efficient computation in the form of analog computing. This manuscript is prepared to support the open-sourced PhyCV code which is available in the GitHub repository: https://github.com/JalaliLabUCLA/phycv''')

st.markdown('**Keywords:** *Streamlit*, *machine learning*, *data science*, *Python*, *PhyCV*, *Physics-inspired Computer Vision Library*')

st.header('Introduction')
st.markdown('''
PhyCV is a new class of computer vision algorithms inspired by physics, the current release of PhyCV has three algorithms: Phase-Stretch Transform (PST) [1], Phase-Stretch Adaptive Gradient-Field Extractor (PAGE) [2, 3], and Vision Enhancement via Virtual diffraction and coherent Detection (VEViD) [4]. These algorithms are originated from the research on photonic time stretch, which is a hardware technique for ultrafast and single-shot data acquisition that exploits dispersion and, in its full-field form, coherent detection [5, 6, 7, 8]. The algorithms emulate the propagation of light through a 2D physical medium with natural and artificial diffractive properties followed by coherent (phase) detection. The diffractive medium will apply a phase kernel to the frequency domain of the image and convert a realvalued image into a complex function. After coherent detection, the output phase contains useful features of the input image. In other words, PhyCV leverages the knowledge of optical physics and adapts it to computational imaging [9]. It is important to note that in PhyCV algorithms, the phase induced by propagation is very small (≤2𝜋). 

''')

image = Image.open('image/image1.png')

st.image(image, caption='Figure 1: A conceptual diagram of PhyCV. The input image is transformed by physics equations in a physical system followed by coherent (phase) detection. The detected phase at the output contains useful features of the input image.')

st.header('From Optical Physics to Algorithms')

st.markdown('''
Photonic time stretch can be understood by considering the propagation of an optical pulse, on which the information of interest has been encoded, through a dispersive optical element. To outline how PhyCV emerged as an offshoot of photonic time stretch, we start with the Nonlinear Schrödinger Equation (NLSE) – the master equation that describes optical pulse propagation through an optical medium. The NLSE contains terms describing attenuation, dispersion and nonlinearity as shown below [10]:
''')

st.latex(r'''\frac{\delta E(t,z)}{\delta z} = \frac{\alpha}{2} E(t,z)- i \frac{\beta_2}{2}  \frac{\delta ^2E(t,z)}{\delta t^2} + i\gamma|E(t,z)|^2E(t,z)''') 

st.markdown('''
By disregarding the loss and nonlinearity (the first and the third terms), the NLSE can be simplified to the integral equation as shown below: 
''')

st.latex(r'''E_0(t) = \frac{1}{2\pi} \int \tilde{E_i}(w) .e^{i\phi(w)} .e^{iwt} dw''')

st.markdown('''In photonic time-stretch systems, the information is encoded onto the spectrum of the input femtosecond pulse 𝐸̃𝑖(𝜔), through propagation, the spectrum is reshaped into a temporal signal at arbitrary time scale with a complex envelope 𝐸𝑜(𝑡) as shown in [6, 8]. 

Next, we convert the operation into discrete domain and introduce the 1D discrete stretch operator 𝕊: 
''')

st.latex(r'''\mathbb{S}\{ E_i[n] \} = 1FFT\{ FFT \{ E_i[n]  \} .\widetilde{K}(K_n).\widetilde{L}(K_n)\}''')


st.markdown('''Here, instead of using continuous variable 𝑡 and 𝜔, we use discrete variable 𝑛 and the corresponding frequency variable 𝑘𝑛. The 1D discrete stretch operator 𝕊 can be written as the input signal spectrum multiplied by some general phase kernel 𝐾̃(𝑘𝑛) which is caused by dispersion or diffraction, and amplitude kernel 𝐿̃(𝑘𝑛). Next, we extend the stretch operator 𝕊 to 2D discrete domain: 
''')

st.latex(r'''\mathbb{S}\{ E_i[m, n] \} = 1FFT^2\{ FFT^2 \{ E_i[m, n]  \} .\widetilde{K}(K_m, K_n).\widetilde{L}(K_m, K_n)\}''')

st.markdown('''Here 𝑚 and 𝑛 are spatial coordinates in the digital image, 𝑘𝑚 and 𝑘𝑛 are the corresponding frequency coordinates. This 2D discrete stretch operator emulates an image propagating through a metaphoric diffractive medium just like a laser pulse propagating through a dispersive optical element, which is the central process that governs photonic time stretch. This demonstrates how the algorithms are inspired by optical physics. The output of the 2D discrete stretch operator 𝕊 is a complex-valued function. After doing coherent detection by mixing the output signal with a local oscillator (LO), the phase which contains useful feature of the input image is detected. 
''')

st.header('Algorithms in PhyCV ')

st.markdown('''In this section, we introduce the three algorithms in the current release of PhyCV: Phase-Stretch Transform (PST), Phase-Stretch Adaptive GradientField Extractor (PAGE), and Vision Enhancement via Virtual diffraction and coherent Detection (VEViD).
''')

st.subheader('''  Phase-Stretch Transform (PST) ''')

st.markdown('''PST is a computationally efficient edge and texture detection algorithm with exceptional performance in visually impaired images [1, 11]. The mathematical operation of PST is shown below:  
''')

st.latex(r'''\mathbb{S}\{ E_i[m, n] \} = 1FFT^2\{ FFT^2 \{ E_i(m, n)  \} .\widetilde{K}(K_m, K_n).\widetilde{L}(K_m, K_n)\}''')

st.latex(r'''PST\{ E_i[m,n] \} = \Im\{ \mathbb{S}\{ E_i[m, n] \}\}''')

st.markdown(''' Here 𝐸𝑖(𝑚,𝑛) is the input image, 𝑚 and 𝑛 are spatial coordinates, 𝑘𝑚 and 𝑘𝑛 are corresponding frequency coordinates. 𝐿̃(𝑘𝑚,𝑘𝑛) is a Gaussian lowpass filter in the frequency domain for image denoising, 𝐾̃(𝑘𝑚,𝑘𝑛)=𝑒−𝑖𝜙(𝑘𝑚,𝑘𝑛) is a nonlinear frequency-dependent phase filter which applies higher amount of phase to higher frequency features of the image. Since sharp transitions, such as edges and corners, contain higher frequencies, by detecting the phase of the output, PST extracts the edge information. The extracted edges are further enhanced by thresholding and morphological operations. In the implementation, we use a phase kernel 𝐾̃(𝑘𝑚,𝑘𝑛) with a phase profile 𝜙(𝑘𝑚,𝑘𝑛) that is symmetric in the polar coordinates as described below: 
''')

st.latex(r'''\phi(k_m,k_n) = \phi_polar (r,\theta) = \phi(r)''')

st.markdown(''' To create a low dimensional phase kernel with the required properties to perform edge and texture detection of the image, we choose the phase profile which has the derivative equals to the inverse tangent function: 
''')

st.latex(r'''\frac {d\phi(r)}{dr} = \tan^{-1}(r)''') 

st.latex(r'''\phi(r) = r.tan^{-1}(r) - \frac 1 2 log(r^2 +1)''') 

st.markdown(''' Therefore, the PST kernel is implemented as: 
''')

st.latex(r'''\phi(k_m,k_n) = \phi(r)''')

st.latex(r'''S. \frac {Wr.tan^{-1}(Wr)- \frac 1 2 log(1+(Wr)^2)} {W_{rmax}.\tan^{-1}(W_{rmax})-\frac 1 2 log(1+ (W_{rmax})^2)}''')

st.markdown(''' PST can also be implemented with other phase kernels. The necessary properties of the kernel have been described in [11]. 

PST has been applied to various tasks including improving the resolution of MRI image [12], extracting blood vessels in retina images to identify various diseases [13], detection of dolphins in the ocean [14], waste water treatment [15], single molecule biological imaging [16], and classification of UAV using micro imaging [17]. 

''')

image = Image.open('image/image2.png')

st.image(image, caption='Figure 2: Retina vessel detection using PST in PhyCV.')

st.subheader('''  Phase-Stretch Adaptive Gradient-Field Extractor (PAGE)  ''')

st.markdown(''' PAGE is a physics inspired feature engineering algorithm that computes a feature set comprised of edges at different spatial frequencies (and hence spatial scales) and orientations [2, 3]. Metaphorically speaking, PAGE emulates the physics of birefringent (orientation-dependent) diffractive propagation through a physical medium with a specific diffractive structure. The mathematical operation of PAGE is shown below: 
''')

st.latex(r'''\mathbb{S}\{ E_i[m, n]; \theta \} = 1FFT^2\{ FFT^2 \{ E_i[m, n]  \} .\widetilde{K}(K_m, K_n; \theta).\widetilde{L}(K_m, K_n)\}''')

st.latex(r'''PAGE\{ E_i[m, n]; \theta \} = \Im\{ \mathbb{S}\{ E_i[m, n]; \theta \}\}''')

st.markdown(''' Here 𝐸𝑖(𝑚,𝑛) is the input image and 𝐿̃(𝑘𝑚,𝑘𝑛) is the denoising filter. Then instead of having one phase filter, PAGE has the phase filter bank 𝐾̃(𝑘𝑚,𝑘𝑛;𝜃), which contains filters with different angle variable 𝜃 that controls the directionality of the detected edge. A change of basis leads to the transformed frequency variables 𝑘𝑚′ and 𝑘𝑛′: 
''')

st.latex(r'''k'_m = k_m.\cos\theta + k_n.\sin\theta''')

st.latex(r'''k'_n = k_m.\sin\theta + k_n.\cos\theta''')

st.markdown(''' The PAGE kernel 𝐾̃(𝑘𝑚,𝑘𝑛;𝜃) now becomes 𝐾̃(𝑘𝑚′,𝑘𝑛′) and it is expressed as a product of two phase functions, 𝜙1 and 𝜙2. The first component 𝜙1 is a symmetric gaussian filter that selects the spatial frequency range of the edges that are detected. Default center frequency is 0, which indicates a baseband filter, the center frequency and bandwidth of which can be changed to probe edges with different sharpness. In other words, it enables the filtering of edges occurring over different spatial scales. The second component, 𝜙2, performs the edge-detection. The explanation of the parameters can be found in [2]. 
''')

st.latex(r'''\widetilde{k}(k_m,k_n; \theta) = \widetilde{k}(k'_m,k'_n) = exp(-i. \phi_1(k'_m).\phi_2(k'_n)''')

st.latex(r'''\phi_1(k'_m) = S_1. \frac{1}{\sqrt{2\pi}\sigma_1} .exp \left( \frac{(|k'_m| - \mu_1)^2}{2 \sigma_1^2} \right)''')

st.latex(r'''\phi_2(k'_n) = S_2. \frac{1}{\sqrt{2\pi}|k_n'|\sigma_2} .exp \left( \frac{(ln|k'_n| - \mu_2)^2}{2 \sigma_2^2} \right)''')

image = Image.open('image/image3.png')

st.image(image, caption='Figure 3: Directional edge detection of a sunflower using PAGE in PhyCV. For visualization, the directions of the edges are mapped into colors.')

st.subheader('''  Vision Enhancement via Virtual diffraction and coherent Detection (VEViD)   ''')

st.markdown(''' VEViD is an efficient and interpretable low-light and color enhancement algorithm that reimagines a digital image as a spatially varying metaphoric light field and then subjects the field to the physical processes akin to diffraction and coherent detection [4]. The term “virtual” captures the deviation from the physical world. The light field is “pixelated” and the propagation imparts a phase with an arbitrary dependence on frequency which can be different from the quadratic behavior of physical diffraction. The mathematical operation of VEViD is shown below: 
''')

st.latex(r'''\mathbb{S}\{ E_i[m, n; c] \} = 1FFT^2\{ FFT^2 \{ E_i(m, n;c) + b  \} .\widetilde{K}(K_m, K_n).\widetilde{L}(K_m, K_n)\}''')

st.latex(r'''VEViD\{ E_i[m, n; c] \} = \tan_{-1} \left(G. \frac {Im{\{\mathbb{S}\{ E_i[m, n; c] \}}} {\{ E_i[m, n; c] \}} \right)''')

st.markdown(''' Here 𝐸𝑖[𝑚,𝑛;𝑐] is the input digital image and 𝑐 represents the color channel in the HSV color space. VEViD leads to low-light enhancement when operating on V (value) channel as shown in Figure 3 and color enhancement when operating on S (saturation) channel as shown in Figure 4. 𝑏 is a regularization term and 𝐺 is the phase activation gain term. 𝐾̃(𝑘𝑚,𝑘𝑛) is the phase kernel which has a phase profile 𝜙(𝑘𝑚,𝑘𝑛) that follows a Gaussian distribution with zero mean. 
''')

st.latex(r'''\widetilde{k}(k_m,k_n) = exp(-i. \phi_1(k'_m).\phi_2(k'_n))''')

st.latex(r'''\phi(k_m,k_n) = S . exp \left( \frac {k^2_m + k^2_n} {T} \right)''')

st.markdown(''' VEViD can be further accelerated through mathematical approximations that reduce the computation time without appreciable sacrifice in image quality. A closed-form approximation for VEViD which we call VEViD-lite is shown below and can achieve up to 200 FPS for 4K video enhancement. The full derivation of the physical and mathematical principle of VEViD can be found in [4]. 
''')

st.latex(r'''VEViD_lite\{ E_i[m, n; c] \} = \tan_{-1} \left(G. \frac {{\{ E_i[m, n; c] + b \}}} {\{ E_i[m, n; c] \}} \right)''')

st.markdown(''' It is also demonstrated that the VEViD serves as a powerful pre-processing tool that improves neural network based object detector in night-time environments without adding computational overhead and retraining.  
''')

image = Image.open('image/image4.png')

st.image(image, caption='Figure 3: Low-light enhancement using VEViD in PhyCV.')

image = Image.open('image/image5.png')

st.image(image, caption='Figure 4: Color enhancement using VEViD in PhyCV.')

image = Image.open('image/image6.png')

st.image(image, caption='Figure 5: The performance of YOLO-v3 object detector is improved by VEViD in PhyCV.')

st.header('''  PhyCV Highlights    ''')

st.subheader('''  Modular Code Architecture     ''')

st.markdown(''' The modular code architecture of PhyCV follows the physics behind the algorithm and is therefore more intuitive. Since all algorithms in PhyCV emulate the propagation of the input image through a device with specific diffractive properties, which applies a phase kernel to the frequency domain of the original image. This process has three steps in general, loading the image, initializing the kernel and applying the kernel. In the implementation, each algorithm is represented as a class and each class has methods that simulate the steps described above. This makes the code easy to understand and extend.
''')

image = Image.open('image/image7.png')

st.image(image, caption='Figure 6: The modular code architecture of PhyCV algorithm PST.')

st.subheader('''   GPU Acceleration     ''')

st.markdown(''' PhyCV also supports GPU acceleration. The GPU versions of PhyCV are built on PyTorch and accelerated by CUDA. The GPU compatibility significantly accelerates the algorithms, which is beneficial for real-time video processing and related deep learning tasks. Here we show the comparison of running time of PhyCV algorithms on CPU and GPU for videos at 1080p, 2K and 4K resolutions. For results shown in the table below, the CPU is Intel i9-9900K @ 3.60GHz x 16 and the GPU is NVIDIA TITAN RTX. Note that the PhyCV low-light enhancement operates in the HSV color space, so the running time also includes RGB to HSV conversion. Moreover, for running time using GPUs, we ignore the time of moving data from CPUs to GPUs and count the algorithm operation time only. 
''')

data = {
    'Resolution': ['1080p', '2K', '4K'],
    'CPU': ['550 ms', '1000 ms', '2290 ms'],
    'GPU': ['4.6 ms', '8.2 ms', '18.5 ms']
}

st.table(data)
st.caption('Table 1. Running time (per frame) of PhyCV – PST edge detection on videos at different resolutions.')

data = {
    'Resolution': ['1080p', '2K', '4K'],
    'CPU': ['2800 ms' , '5000 ms', '11660 ms'],
    'GPU': ['48.5 ms' , '87 ms', '197 ms']
}

st.table(data)
st.caption('Table 2. Running time (per frame) of PhyCV – PAGE directional edge detection on videos at different resolutions.')

data = {
    'Resolution': ['1080p', '2K', '4K'],
    'CPU': ['175 ms' , '320 ms', '730 ms'],
    'GPU': ['4.3 ms' , '7.8 ms', '17.9 ms']
}

st.table(data)
st.caption('Table 3. Running time (per frame) of PhyCV – VEViD lowlight enhancement on videos at different resolutions. RGB to HSV conversion time is included.')

data = {
    'Resolution': ['1080p', '2K', '4K'],
    'CPU': ['60 ms' , '110 ms', '245 ms'],
    'GPU': ['2.1 ms' , '3.5 ms', '7.4 ms']
}

st.table(data)
st.caption('Table 4. Running time (per frame) of PhyCV – VEViD-lite low-light enhancement on videos at different resolutions. RGB to HSV conversion time is included.')

st.header('Reference Link')
st.markdown('''
**A Report on PhyCV: The First Physics-inspired Computer Vision Library** : https://arxiv.org/ftp/arxiv/papers/2301/2301.12531.pdf

**Repository link** : https://github.com/JalaliLabUCLA/phycv
''')

st.header('References')
with st.expander("Expand"):
    st.markdown(''' [1]  M. Asghari and B. Jalali, "Edge detection in digital images using dispersive phase stretch transform," International journal of biomedical imaging, 2015.

    [2]  C. MacPhee, M. Suthar and B. Jalali, "PhaseStretch Adaptive Gradient-Field Extractor (PAGE)," arXiv preprint arXiv:2202.03570., 2022.

    [3]  M. Suthar and B. Jalali, "Phase-stretch adaptive gradient-field extractor (page)," in Coding Theory, 2020.  

    [4]  B. Jalali and C. MacPhee, "VEViD: Vision Enhancement via Virtual diffraction and coherent Detection," eLight, 2022.  

    [5]  A. Bhushan, F. Coppinger and B. Jalali, "Timestretched analogue-to-digital conversion," Electronics Letters, 1998.  

    [6]  A. Fard, S. Gupta and B. Jalali, "Photonic timestretch digitizer and its extension to real‐time spectroscopy and imaging," Laser & Photonics Reviews, 2013.  

    [7]  A. Mahjoubfar, D. Churkin, S. Barland, N. Broderick, S. Turitsyn and B. Jalali, "Time stretch and its applications," Nature Photonics, 2017.  

    [8]  Y. Zhou, J. Chan and B. Jalali, "A Unified Framework for Photonic Time‐Stretch Systems," Laser & Photonics Reviews, 2022.  

    [9]  B. Jalali, M. Suthar, M. Asghari and A. Mahjoubfar, "Physics-based feature engineering," in Optics, Photonics and Laser Technology, 2019.  

    [10]  G. Agrawal, Nonlinear fiber optics, 2000.  

    [11]  M. Suthar, H. Asghari and B. Jalali, "Feature enhancement in visually impaired images," IEEE Access, 2017.  

    [12]  S. He and B. Jalali, "Fast super-resolution in MRI images using phase stretch transform, anchored point regression and zero-data learning," in IEEE International Conference on Image Processing (ICIP), 2019 .  

    [13]  M. Challoob and Y. Gao, "A local flow phase stretch transform for robust retinal vessel detection," in International Conference on Advanced Concepts for Intelligent Vision Systems, 2020.  

    [14]  S. Wang, Z. Cai, W. Cao and J. Yuan, "Dolphin Identification Method Based on Improved PST," in IEEE/ACIS 6th International Conference on Big Data, Cloud Computing, and Data Science (BCD), 2021.  

    [15]  R. Ang, H. Nisar, M. Khan and C. Tsai, "Image segmentation of activated sludge phase contrast images using phase stretch transform," Microscopy, 2019.  

    [16]  T. Ilovitsh, B. Jalali, M. Asghari and Z. Zalevsky, "Phase stretch transform for superresolution localization microscopy," Biomedical optics express, 2016.  

    [17]  A. Singh and Y. Kim, "Classification of Drones Using Edge-Enhanced Micro-Doppler Image Based on CNN," Traitement du Signal, 2021.     

    ''')


hide_streamlit_style = '''
            <style>
            footer {visibility: hidden;}
            </style>
            '''
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

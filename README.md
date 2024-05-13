                                                                                                                     
                                   ,,                                                 ,,    ,,                       
`7MMF'    `7MMF'   `7MF'         `7MM                                               `7MM    db                       
  MM        MM       M             MM                                                 MM                             
  MM        MM       M        ,M""bMM  .gP"Ya   ,p6"bo   ,pW"Wq.`7MM  `7MM `7MMpdMAo. MM  `7MM  `7MMpMMMb.  .P"Ybmmm 
  MM        MM       M      ,AP    MM ,M'   Yb 6M'  OO  6W'   `Wb MM    MM   MM   `Wb MM    MM    MM    MM :MI  I8   
  MM      , MM       M      8MI    MM 8M"""""" 8M       8M     M8 MM    MM   MM    M8 MM    MM    MM    MM  WmmmP"   
  MM     ,M YM.     ,M      `Mb    MM YM.    , YM.    , YA.   ,A9 MM    MM   MM   ,AP MM    MM    MM    MM 8M        
.JMMmmmmMMM  `bmmmmd"'       `Wbmd"MML.`Mbmmd'  YMbmd'   `Ybmd9'  `Mbod"YML. MMbmmd'.JMML..JMML..JMML  JMML.YMMMMMb  
                                                                             MM                            6'     dP 
                                                                           .JMML.                          Ybmmmd'   

Pre-requisite Python libraries: Matplotlib, SciPy, SymPy, NumPy
This repository is to help you numerically decouple systems of coupled differential equations with complicated 
and/or oscillatory behaviour. I have not had time to fully clean the code, but it is well-commented and optimised 
for use. The code presented was designed for a Part III research project at the Kavli Institute for Cosmology,
Cambridge. 

LU decomposition.py
This numerically decouples a reduced version of the Einstein-Boltzmann matrix equations and then verifies the
solution with RK4(5) integration. This is then exported as LUexample.pdf. The cosmological equations used 
here assume radiation dominance in the early universe, but you can adapt this as needed. 

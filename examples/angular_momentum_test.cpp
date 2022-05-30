//*******************************************************************************************
//	testing "amo".
//
//	by:  Mohammed Maher Abdelrahim Mohammed
//	     UNIVERSITÀ DELLA CALABRIA, DIPARTIMENTO DI FISICA AND INFN-COSENZA
//	     VIA P. BUCCI, CUBO 31 C, I-87036 COSENZA, ITALY
//	     mohammed.maher@unical.it                                          
//*******************************************************************************************
#include<iostream>
#include<complex>
#include<ctime>
#include"utilities.h"
#include"operators.h"
#include"braket.h"

	typedef std::complex<double>  complex;
	using std::cout;
	using std::endl;
 
int main()
{
        ///<--clock stuff
	std::clock_t start;
	double duration;
 	start = std::clock();
 	///<--stop clock stuff
 	
 	/*
 	double spin; 
 	cout<<"Insert valid spin: (e.g 1/2, 3/2, 5/2,...) \n";
 	std::cin>>spin;
 	*/
        const double spin= 1./2.; // 3./2.;
	 
	
	AngularMomentum<complex> S_x, S_y, S_z;
	AngularMomentum<complex> S_sqr, S_plus, S_minus;
       
	S_sqr = S_sqr.AngularMomentum_JSquare(spin);
	S_x = S_x.AngularMomentum_Jx(spin);
	S_y = S_y.AngularMomentum_Jy(spin);
	S_z = S_z.AngularMomentum_Jz(spin);
	S_plus = S_plus.AngularMomentum_JPlus(spin);
	S_minus = S_minus.AngularMomentum_JMinus(spin);

	cout<<"For spin ";
	DecimalToFraction(spin);
	cout<<" system:\n";

	cout<<"------------S_x operator:-------------\n";
	ResultPrint(S_x);

	cout<<"------------S_y operator:-------------\n";
	ResultPrint(S_y);
	cout<<"------------S_z operator:-------------\n";
	ResultPrint(S_z);

	cout<<"------------S^2 operator:-------------\n";
	ResultPrint(S_sqr);

	cout<<"------------S_+ operator:-------------\n";
	ResultPrint(S_plus);
	cout<<"------------S_- operator:-------------\n";
	ResultPrint(S_minus);
	
	///<--clock stuff again
	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
 
	if (duration < 60.0) {
		std::cout << "Elapsed time: " << duration << " seconds" << std::endl;
	}
 
	else {
		std::cout << "Elapsed time: " << duration/60.0 << " mins" << std::endl;
	}
	
	return 0;
}

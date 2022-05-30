/*###########################################################################
  File:      BraKET.H

  Summary:   Define and handle some fundamental operations on Bra and 
  	     ket vectors.
  	     
  Classes:   ket and Bra.  

  by:  	     Mohammed Maher Abdelrahim Mohammed
       	     UNIVERSITÀ DELLA CALABRIA, DIPARTIMENTO DI FISICA AND INFN-COSENZA
	     VIA P. BUCCI, CUBO 31 C, I-87036 COSENZA, ITALY
       	     mohammed.maher@unical.it  
###########################################################################*/
//#pragma once
#ifndef BRAKET_H
#define BRAKET_H
#include <Eigen/Dense>
#include <algorithm>
#include <cstdarg>

//namespace QuantaPlus {

/***************************************************************************
  Class:    Ket.

  Summary:  Ket is a public derived from the well designed Eigen::Matrix class. 
	    Ket vectors represent matrices with one column and Dynamic-size rows,

  Methods:  MyMethodOne
              Short description of MyMethodOne.
            MyMethodTwo
              Short description of MyMethodTwo.
            Ket
              Constructor.
            ~Ket
              Destructor.
***************************************************************************/
template <class T>
class Ket : public Eigen::Matrix<T,Eigen::Dynamic,1> 
{
public:
	Ket();
	Ket(int row); 
   	//Ket(const T *data ); 
	~Ket();     
};
/***************************************************************************
  Class:    Bra.

  Summary:  Bra is a public derived from the well designed Eigen::Matrix class. 
	    Bra vectors are matrices with one row, and Dynamic-size column.

  Methods:  MyMethodOne
              Short description of MyMethodOne.
            MyMethodTwo
              Short description of MyMethodTwo.
            Bra
              Constructor.
            ~Bra
              Destructor.
***************************************************************************/
template <class T>
class Bra : public Eigen::Matrix<T,1, Eigen::Dynamic> 
{
public:
	Bra();
	Bra(int col); 
   	// Bra(const T *data ); 
	~Bra();  
};
#include"braket.tpp"
//} //end of namespace QuantaPlus
#endif

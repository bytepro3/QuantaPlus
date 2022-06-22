#include<iostream>
#include<cmath>
#include"utilities.h"
#include"operators.h"
#include<vector>
#include<list>
#include <functional>
using namespace std;

struct ketstate{
	double J1,M1,J2,M2;
	double factor=1.0;
	ketstate(){}
	ketstate(const double& j1, const double& m1,const double& j2,  const double& m2)
			:J1(j1),M1(m1),J2(j2),M2(m2){}
	void PrintF(){
			if(factor==1.0){
				cout<<"AMuket["<<J1<<", "<<M1<<", "<<J2<<", "<<M2<<"]\n";
			}
			else{
				DecimalToFraction(factor);cout<<" AMuket["<<J1<<", "<<M1<<", "<<J2<<", "<<M2<<"]\n";
			}   
		}

};
template<typename T>
int Sign(T& n){
 return ( n >= 0 ) ? 1: -1;
}
//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
class Basis{
	public:
		double J1,M1,J2,M2;
		double factor=1.0;
		std::vector<Basis> Basis_list;
//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
		Basis(){}
//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
		Basis(const double& j1, const double& j2):J1(j1),J2(j2) {
			setBasis_list();
		}
		Basis(const double& j1, const double& m1,const double& j2,  const double& m2)
			:J1(j1),M1(m1),J2(j2),M2(m2) {
		
		}
//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
		void PrintF(){
			if(factor==1.0){
				cout<<"AMuket["<<J1<<", "<<M1<<", "<<J2<<", "<<M2<<"]\n";
			}
			else if(factor==0){cout<<0<<"\n";}
			else{
				DecimalToFraction(factor);cout<<" AMuket["<<J1<<", "<<M1<<", "<<J2<<", "<<M2<<"]\n";
				//cout<<factor<<" AMuket["<<J1<<", "<<M1<<", "<<J2<<", "<<M2<<"]\n";
			}   
		}
//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
		void setBasis_list(){
			for(auto i=0.; i<=int(2);i+=J1){
				for(auto j=0.; j<=int(2);j+=J2){
					Basis_list.push_back( Basis(J1,J1-i,J2,J2-j) );
				}
			}
		}
//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
		void PrintFBasis_List(){ 
			for (auto n : Basis_list) {
				n.PrintF();
			}
			std::cout <<"\n";
		}
//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
		struct Jop{ Jop(); };

		friend int operator* (const Basis& b1 , const Basis& b2);
	
};

//------------------------------------------------------------------------
//AMuket /: AMuket[j1_, m1_, j2_, m2_] AMuket[j1p_, m1p_, j2p_, m2p_] = 0; /;
//  (j1 =!= j1p) || (m1 =!= m1p) || (j2 =!= j2p) || (m2 =!= m2p);
//AMuket /: AMuket[j1_, m1_, j2_, m2_] AMuket[j1_, m1_, j2_, m2_] = 1;
//AMuket /: Power[AMuket[j1_, m1_, j2_, m2_], 2] = 1;
//------------------------------------------------------------------------
int operator* (const Basis& b1 , const Basis& b2)
{
     int n=1;
     if((b1.J1!=b2.J1) || (b1.J2!=b2.J2) ||  (b1.M1!=b2.M1) ||  (b1.M2!=b2.M2) ){n=0;}
    
      return n;
}
/*double operator* (const double& num , const Basis& B)
{
      double res;
      res = num*B.basis_list
    
      return n;
}*/
//------------------------------------------------------------------------
//AM /: AM[j12] AMuket[j1_, m1_, j2_, m2_] := j1 (j1 + 1) AMuket[j1, m1, j2, m2];
//------------------------------------------------------------------------
std::vector<Basis> J12(const Basis& b){
       	std::vector<Basis> res;
	for(auto n : b.Basis_list) {
		n.factor=  n.J1*(n.J1+1.);

		res.push_back(n);
	}
     	return res;    
}
//------------------------------------------------------------------------
//AM /: AM[j1z] AMuket[j1_, m1_, j2_, m2_] := m1*AMuket[j1, m1, j2, m2];
//------------------------------------------------------------------------
std::vector<Basis> J1z(const Basis& b){  
    	std::vector<Basis> res;
	for(auto n : b.Basis_list) {
		n.factor=  n.M1;

		res.push_back(n);
	}
     	return res;      
}
//------------------------------------------------------------------------
//AM /: AM[j22] AMuket[j1_, m1_, j2_, m2_] := j2 (j2 + 1) AMuket[j1, m1, j2, m2];
//------------------------------------------------------------------------
std::vector<Basis> J22(const Basis& b){
 
 	std::vector<Basis> res;	
	for(auto n : b.Basis_list) {
		n.factor=  n.J2*(n.J2+1.);

		res.push_back(n);
	}
     	return res;  
     
}
//------------------------------------------------------------------------
//AM /: AM[j2z] AMuket[j1_, m1_, j2_, m2_] := m2*AMuket[j1, m1, j2, m2];
//------------------------------------------------------------------------
std::vector<Basis> J2z(const Basis& b){
 	
	std::vector<Basis> res;
	for(auto n : b.Basis_list) {
		n.factor=  n.M2;

		res.push_back(n);
	}
     	return res;
}
//------------------------------------------------------------------------
//AM /: AM[j1p] AMuket[j1_, m1_, j2_, m2_] := Sqrt[j1 (j1 + 1) - m1 (m1 + 1)] AMuket[j1, m1 + 1, j2, m2];
//------------------------------------------------------------------------
std::vector<Basis> J1p(const Basis& b){
	
	std::vector<Basis> res;
	for(auto n : b.Basis_list) {
		n.factor= std::sqrt( std::abs(((n.J1 - n.M1))*(n.J1+n.M1+1.)) );
		n.M1 +=1.; 

		res.push_back(n);
	}
     	return res; 
}
//------------------------------------------------------------------------
//AM /: AM[j1m] AMuket[j1_, m1_, j2_, m2_] := Sqrt[j1 (j1 + 1) - m1 (m1 - 1)] AMuket[j1, m1 - 1, j2, m2];
//------------------------------------------------------------------------
std::vector<Basis> J1m(const Basis& b){
 
    	std::vector<Basis> res;
	for(auto n : b.Basis_list) {
		n.factor= std::sqrt( std::abs(((n.J1 + n.M1))*(n.J1-n.M1+1.)) );
		n.M1 -=1.; 

		res.push_back(n);
	}
	return res;
}
//------------------------------------------------------------------------
//AM /: AM[j2p] AMuket[j1_, m1_, j2_, m2_] := Sqrt[j2 (j2 + 1) - m2 (m2 + 1)] AMuket[j1, m1, j2, m2 + 1];
//------------------------------------------------------------------------
std::vector<Basis> const J2p(const Basis& b){
 
	std::vector<Basis> res;
	for(auto n : b.Basis_list) {
		n.factor= std::sqrt( std::abs(((n.J2 - n.M2))*(n.J2+n.M2+1.)) );
		n.M2 +=1.; 

		res.push_back(n);
	}
     	return res;   
}
//------------------------------------------------------------------------
//AM /: AM[j2m] AMuket[j1_, m1_, j2_, m2_] := Sqrt[j2 (j2 + 1) - m2 (m2 - 1)] AMuket[j1, m1, j2, m2 - 1];
//------------------------------------------------------------------------
std::vector<Basis> const J2m(const Basis& b){
 
       	std::vector<Basis> res;
	for(auto n : b.Basis_list) {
		n.factor= std::sqrt( std::abs(((n.J2 + n.M2))*(n.J2-n.M2+1.)) );
		n.M2 -=1.; 

		res.push_back(n);
	}
	return res;
}
 

 

//------------------------------------------------------------------------
// AM /: AM[a_, b_] ket_ := AM[a] (Compose @@ Join[AM /@ {b}, {ket}]);
// AM /: AM[a_][ket_] := AM[a] ket;
//------------------------------------------------------------------------
std::vector<double>  OpratorTimesBasis(std::vector<Basis> fun(const Basis&),const Basis& target){
 	
 	std::vector<Basis> pro_list = fun(target);
	std::vector<double> res;
	for(auto m : target.Basis_list){
		for (auto n : pro_list){
			res.push_back(n*m*n.factor);
		}
	}
	return res;
}
  
//------------------------------------------------------------------------
// AMmatrix[op_, basis_List] := Expand[Outer[Times, Expand[op basis], basis]]
//------------------------------------------------------------------------
Eigen::MatrixXd AMmatrix(const std::vector<double>& L, const Basis& B)
{

	int k = 0;
	 
	Eigen::VectorXd v(L.size());
	for (auto n : L) { 
		v(k++)=n;	 
	}
	//std::cout <<"\n";
	
	Eigen::MatrixXd m(B.Basis_list.size(),B.Basis_list.size());	
	for(int i=0; i<B.Basis_list.size(); i++) {
		for(int j=0; j<B.Basis_list.size(); j++){
			m(i,j)=v[j+i*B.Basis_list.size()];
		}
	}

	return m.transpose();
}

Eigen::VectorXd AMvector(const std::vector<double>& L)
{
	int k = 0;
	 
	Eigen::VectorXd v(L.size());
	for (auto n : L) { 
		v(k++)=n;	 
	}

	return v;
}

/*Eigen::VectorXd AMvector( const Basis& B)
{
	int k = 0;
	 
	Eigen::VectorXd v(B.size());
	for (auto n :B.Basis_list) { 
		v(k++)=n;	 
	}

	return v;
}*/
 
//------------------------------------------------------------------------
//setphase[m_?MatrixQ] := Table[Sign[N[Select[m[[i]]]][[1]]], {i, Length[m]}] m;
//------------------------------------------------------------------------
Eigen::MatrixXd setphase(Eigen::MatrixXd& mat){
	Eigen::MatrixXd res(mat.rows(),mat.cols());
	for(int i=0; i<mat.rows();i++){
		for(int j=0; j<mat.cols();j++){

		if(mat(i,j)<0) {res(i,j)=std::abs(mat(i,j));}
		else res(i,j)=mat(i,j);
		}
	}
	return res;
}
//------------------------------------------------------------------------
//jval[j1_, j2_, v_] := Array[(-1 + Sqrt[1 + 4*(j1 (j1 + 1) + j2 (j2 + 1) + 2*v[[#]])])/2 &, Length[v]]
//------------------------------------------------------------------------
Eigen::VectorXd jval(const double& j1, const double& j2, Eigen::VectorXd& vec){
    
      Eigen::VectorXd res(vec.size());
      for(int i =0; i<vec.size(); i++){
      
      res[i] =(-1.+ std::sqrt(1.+4.*(j1*(j1+1.) +j2*(j2+1.)+2.*vec(i))))/2.;
      }
      return res;
      
}
//------------------------------------------------------------------------
// norm[m_?MatrixQ] := Array[m[[#]] . m[[#]] &, Length[m]];
//------------------------------------------------------------------------
//Eigen::MatrixXd Norm(Eigen::MatrixXd& mat){
Eigen::VectorXd Norm(Eigen::MatrixXd& mat){
	//Eigen::MatrixXd res(mat.rows(),mat.cols());
	Eigen::VectorXd res(mat.cols());
	for(int i=0; i<mat.rows();i++){
		for(int j=0; j<mat.cols();j++){
			res(i)+=mat(i,j)*mat(i,j);
		}
	}
	return res;
}
//------------------------------------------------------------------------
// normalize[x_] := x/Sqrt[norm[x]];
//------------------------------------------------------------------------
Eigen::MatrixXd Normalize(Eigen::MatrixXd& mat){
	Eigen::MatrixXd res(mat.rows(),mat.cols());
	Eigen::VectorXd norm_res(mat.cols());
	norm_res = Norm(mat);
	for(int i=0; i<mat.rows();i++){
		for(int j=0; j<mat.cols();j++){
			res(i,j)=mat(i,j)/std::sqrt(norm_res(i));
		}
	}
	return res;
}

/*
std::vector<Basis> AMuJ1J2(const Basis& b){
 
	std::vector<Basis> res;
	
	for(auto n : b.Basis_list) {
		n.factor= std::sqrt( std::abs(((n.J2 - n.M2))*(n.J2+n.M2+1.)) );
		n.M2 +=1.; 

		res.push_back(n);
	}
     	return res;   
}*/
//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
/*std::vector<double> AM(std::vector<Basis>& f1(const Basis&),std::vector<Basis>& f2(const Basis&), const Basis& target){
 std::vector<double> vfirst;
 std::vector<double> vsecond
 vfirst = OpratorTimesBasis(f1(target),target);

}*/
int main(){

	double spin1,spin2;
	
	 
	cout<<"Insert two spins values j1 and j2:\n";
	 cin>>spin1>>spin2;
 
	
	Basis b(spin1,spin2);
	
	//b.PrintFBasis_List();
	vector<double> Basislist = OpratorTimesBasis(J1p,b);
	 
	/*vector<double> vop1 = OpratorTimesBasis(J1z,b);
	vector<double> vop2 = OpratorTimesBasis(J2z,b);
	
	vector<double> sum(vop1.size());
	int product=b*b;
	for(int i=0; i<vop1.size();i++){

	 sum[i]=(vop1[i]+vop2[i]);
	
	}
	
	for(int i=0; i<sum.size();i++){

	 cout<<sum[i]<<", ";
	
	}*/
	//cout<<endl;
	
	
	//Eigen::MatrixXd A1 = AMmatrix(sum, b); 
	//Eigen::MatrixXd A2 = AMmatrix(sum, b);
	cout<<AMmatrix(Basislist, b); 
	//cout<<endl;
	//cout<<AMvector(vop1)<<"\t";
	//cout<<A1<<endl;
	
	/*Eigen::VectorXd A(9);
	 A<<-2,-1,-1,-1,1,1,1,1,1;
	// cout<<A<<endl;
	// cout<<setphase(A)<<endl;
       // cout<<jval(1,1,A)<<endl;
 	//Eigen::VectorXd A(8);
 
 	cout<<jval(1,1,A)<<endl;*/
      
      /*  Eigen::MatrixXd M(3,3);
        M<<1,2,3,4,5,6,7,8,9;
        cout<<Normalize(M)<<endl;*/
 
         
	return 0;
}

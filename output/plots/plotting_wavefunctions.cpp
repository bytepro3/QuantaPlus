#include<iostream>
#include"plt_all.h"
//#include"plot.h"
int main()
{
	plot_phi0();
	plot_phi1();
	plot_phi2();
	plot_phi3();	
	plot_phi();
	DoubleWellplot_phi01();
	DoubleWellplot_phi67();	
	
	//TODO not working...   
	/*Plot p("phi0","x","$\\phi$","output/phi0.dat", "-8:8", "0:1");
	p.savefig("myplot.pdf");
	p.show();*/
}

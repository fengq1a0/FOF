#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include"Eigen/Eigen"
#include"Eigen/IterativeLinearSolvers"
#include<vector>


#include<iostream>
namespace py = pybind11;

void smooth(py::array_t<float> _v,
            py::array_t<int> _f,
            py::array_t<float> _not_z)
{
    auto v = _v.mutable_unchecked<2>();
    auto f = _f.unchecked<2>();
    auto not_z = _not_z.unchecked<1>();

    std::vector<int> a2b(v.shape(0)), b2a(v.shape(0));
    int cnt = 0;
    for (int i=0;i<v.shape(0);i++)
    if (not_z[i])
    {
        b2a[cnt] = i;
        a2b[i] = cnt++;
    }

    std::vector<std::vector<int> > g(v.shape(0));
    for (int i=0;i<f.shape(0);i++)
    {
        g[f(i,0)].push_back(f(i,1));
        g[f(i,1)].push_back(f(i,2));
        g[f(i,2)].push_back(f(i,0));
    }

	Eigen::MatrixXf B = Eigen::MatrixXf::Zero(v.shape(0), 3);
	Eigen::MatrixXf X = Eigen::MatrixXf::Zero(cnt, 3);
    Eigen::SparseMatrix<float> A(v.shape(0), cnt);

    // set A & B
	std::vector<Eigen::Triplet<float>> tlist;
	for (int i = 0; i < v.shape(0); i++)
	{
		if (not_z[i])
			tlist.push_back(Eigen::Triplet<float>(i, a2b[i], -1));
		else
		{
			B(i, 0) += v(i,0);
			B(i, 1) += v(i,1);
			B(i, 2) += v(i,2);
		}

		for (int j = 0; j < g[i].size(); j++)
		{
			int t = g[i][j];
			float val = 1.0 / g[i].size();
			if (not_z[t])
				tlist.push_back(Eigen::Triplet<float>(i, a2b[t], val));
			else
			{
				B(i, 0) -= v(t,0) * val;
				B(i, 1) -= v(t,1) * val;
				B(i, 2) -= v(t,2) * val;
			}	
		}
	}
	A.setFromTriplets(tlist.begin(), tlist.end());
	// set X
	for (int i = 0; i < cnt; i++)
	{
		X(i, 0) = v(b2a[i],0)*1024;
		X(i, 1) = v(b2a[i],1)*1024;
		X(i, 2) = v(b2a[i],2)*1024;
	}
	// go
	{
		Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper,
			Eigen::IncompleteCholesky<float> > solver;
		solver.compute(A.transpose()*A);
		X = solver.solveWithGuess(A.transpose()*B*1024, X);
	}

	// output
	for (int i = 0; i < cnt; i++)
	{
		v(b2a[i],0) = X(i, 0)/1024;
		v(b2a[i],1) = X(i, 1)/1024;
		v(b2a[i],2) = X(i, 2)/1024;
	}
}

void get_mat(py::array_t<float> _v,
             py::array_t<int> _f,
             py::array_t<float> _not_z)
{
    auto v = _v.unchecked<2>();
    auto f = _f.unchecked<2>();
    auto not_z = _not_z.unchecked<1>();

    std::vector<int> a2b(v.shape(0)), b2a(v.shape(0));
    int cnt = 0;
    for (int i=0;i<v.shape(0);i++)
    if (not_z[i])
    {
        b2a[cnt] = i;
        a2b[i] = cnt++;
    }

    std::vector<std::vector<int> > g(v.shape(0));
    for (int i=0;i<f.shape(0);i++)
    {
        g[f(i,0)].push_back(f(i,1));
        g[f(i,1)].push_back(f(i,2));
        g[f(i,2)].push_back(f(i,0));
    }

	
	Eigen::MatrixXf B = Eigen::MatrixXf::Zero(v.shape(0), 3);
	Eigen::MatrixXf X = Eigen::MatrixXf::Zero(cnt, 3);
    Eigen::SparseMatrix<float> A(v.shape(0), cnt);

    // set A & B
	std::vector<Eigen::Triplet<float>> tlist;
	for (int i = 0; i < v.shape(0); i++)
	{
		if (not_z[i])
			tlist.push_back(Eigen::Triplet<float>(i, a2b[i], -1));
		else
		{
			B(i, 0) += v(i,0);
			B(i, 1) += v(i,1);
			B(i, 2) += v(i,2);
		}

		for (int j = 0; j < g[i].size(); j++)
		{
			int t = g[i][j];
			float val = 1.0 / g[i].size();
			if (not_z[t])
				tlist.push_back(Eigen::Triplet<float>(i, a2b[t], val));
			else
			{
				B(i, 0) -= v(t,0) * val;
				B(i, 1) -= v(t,1) * val;
				B(i, 2) -= v(t,2) * val;
			}	
		}
	}
	A.setFromTriplets(tlist.begin(), tlist.end());
	// set X
	for (int i = 0; i < cnt; i++)
	{
		X(i, 0) = v(b2a[i],0);
		X(i, 1) = v(b2a[i],1);
		X(i, 2) = v(b2a[i],2);
	}

}


PYBIND11_MODULE(solver, m) {
    m.def("smooth", &smooth,
        py::arg("_v").noconvert(),
        py::arg("_f").noconvert(),
        py::arg("_not_z").noconvert()
    );
}
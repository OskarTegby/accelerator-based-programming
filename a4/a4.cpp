/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

void checkSizes( int &N, int &nrepeat );

int main( int argc, char* argv[] )
{
  int N = -1;         // number of matrices
  int nrepeat = 100;  // number of repeats of the test

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Matrices" ) == 0 ) ) {
      N = pow( 2, atoi( argv[ ++i ] ) );
      printf( "  User N is %d\n", N );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
      printf( "  User nrepeat is %d\n", nrepeat );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "   Options:\n" );
      printf( "  -Matrices (-N) <int>:  exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( N, nrepeat );

  Kokkos::initialize( argc, argv );
  {

  // Allocate A and J on device.
  typedef Kokkos::View<double*[3][3]> ViewVectorType3;
  typedef Kokkos::View<double*[4][4]> ViewVectorType4;
  
  ViewVectorType4 A( "A", N );
  ViewVectorType3 J( "J", N );

  // Initialize J matrices on host.
  for ( int i = 0; i < N; ++i ) {
    J ( i, 0, 0 ) = 3;  J ( i, 0, 1 ) = 1;  J ( i, 0, 2 ) = 1;
    J ( i, 1, 0 ) = 1;  J ( i, 1, 1 ) = 3;  J ( i, 1, 2 ) = 1;
    J ( i, 2, 0 ) = 1;  J ( i, 2, 1 ) = 1;  J ( i, 2, 2 ) = 3;
  }

  // Timer products.
  Kokkos::Timer timer;

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {

    Kokkos::parallel_for("fem", N, KOKKOS_LAMBDA ( int i ) {

      for ( int i = 0; i < N; ++i ) {
        double C0 = J(i, 1, 1) * J(i, 2, 2) - J(i, 1, 2) * J(i, 2, 1);
        double C1 = J(i, 1, 2) * J(i, 2, 0) - J(i, 1, 0) * J(i, 2, 2);
        double C2 = J(i, 1, 0) * J(i, 2, 1) - J(i, 1, 1) * J(i, 2, 0);
        double inv_J_det = J(i, 0, 0) * C0 + J(i, 0, 1) * C1 + J(i, 0, 2) * C2;
        double d = (1./6.) / inv_J_det;
        double G0 = d * (J(i, 0, 0) * J(i, 0, 0) + J(i, 1, 0) * J(i, 1, 0) + J(i, 2, 0) * J(i, 2, 0));
        double G1 = d * (J(i, 0, 0) * J(i, 0, 1) + J(i, 1, 0) * J(i, 1, 1) + J(i, 2, 0) * J(i, 2, 1));
        double G2 = d * (J(i, 0, 0) * J(i, 0, 2) + J(i, 1, 0) * J(i, 1, 2) + J(i, 2, 0) * J(i, 2, 2));
        double G3 = d * (J(i, 0, 1) * J(i, 0, 1) + J(i, 1, 1) * J(i, 1, 1) + J(i, 2, 1) * J(i, 2, 1));
        double G4 = d * (J(i, 0, 1) * J(i, 0, 2) + J(i, 1, 1) * J(i, 1, 2) + J(i, 2, 1) * J(i, 2, 2));
        double G5 = d * (J(i, 0, 2) * J(i, 0, 2) + J(i, 1, 2) * J(i, 1, 2) + J(i, 2, 2) * J(i, 2, 2));

        A(i, 0, 0) = G0;
        A(i, 0, 1) = A(i, 1, 0) = G1;
        A(i, 0, 2) = A(i, 2, 0) = G2;
        A(i, 0, 3) = A(i, 3, 0) = -G0 - G1 - G2;
        A(i, 1, 1) = G3;
        A(i, 1, 2) = A(i, 2, 1) = G4;
        A(i, 1, 3) = A(i, 3, 1) = -G1 - G3 - G4;
        A(i, 2, 2) = G5;
        A(i, 2, 3) = A(i, 3, 2) = -G2 - G4 - G5;
        A(i, 3, 3) = G0 + 2 * G1 + 2 * G2 + G3 + 2 * G4 + G5;
      }

    });

  }

  // Calculate time.
  double time = timer.seconds();

  // Calculate bandwidth.
  // The A matrix (of size 16) is read N times.
  // The J matrix (of size 9) is read N times.
  double Gbytes = 1.0e-9 * double( sizeof(double) * N * (9 + 16) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N = %d, nrepeat = %d, data size = %g MB, time = %g s, bandwidth = %g GB/s\n",
          N, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

  }
  Kokkos::finalize();

  return 0;
}

void checkSizes( int &N, int &nrepeat ) {

  // If N is undefined, set it.
  if ( N == -1 ) N = 4096;

  // If nrepeat is undefined, set it.
  if ( nrepeat == -1 ) nrepeat = 100;

  // Check sizes.
  if ( ( N < 0 ) || ( nrepeat < 0 ) ) {
    printf( "  N and nrepeat must be greater than 0.\n" );
    exit( 1 );
  }

}

function [ N ] = VC_Nsolver( error_deviation, confidence, dVC )
% Returns expression N. Use eval afterwards.
syms N;
delta = 1-confidence;
N = solve(error_deviation == sqrt((log((2*N*exp(1) / dVC)^dVC * (2/delta)))/(2*N)), N) ;
end


%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Static  Analysis for isotropic materials by FEM.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

%% ---------------------------------------
clear all; clc; close all;

%% GLOBAL VARIABLES
global node element 

%% Geometry
global L D
L = 0.2;
D = 0.1;

%% Material properties: isotropic 
global E nu C stressState
E  = 2e11 ;
nu = 0.3 ;
stressState = 'PLANE_STRESS';

if ( strcmp(stressState,'PLANE_STRESS') )
    C = E/(1-nu^2)*[ 1   nu 0;
        nu  1  0 ;
        0   0  0.5*(1-nu) ];
else
    C = E/(1+nu)/(1-2*nu)*[ 1-nu  nu  0;
        nu    1-nu 0;
        0     0  0.5-nu ];
end

%% Loading
global sigmato
sigmato = -20e6;

%% Meshing
elemType = 'Q4' ;
node = [0 0; L 0; 2*L 0; 2*L D; L D; 0 D];
element = [1 2 5 6; 2 3 4 5];

%% compute number of nodes, of elements, boundaries
numNode = size(node,1);
numElem = size(element,1);

% define essential boundaries
leftNodes = [1; 6];
topNodes = [5; 6];

%% Plot mesh
figure
hold on
axis equal
scatter(node(:,1),node(:,2),5);
plot(node(leftNodes,1),node(leftNodes,2),'ks');
for i = 1:numNode
   text(node(i,1)+L/20,node(i,2)+L/20,num2str(i));    
end

%% Initialize stiffness matrix, force vector

tDof = numNode*2; % 
K = zeros(tDof,tDof); % stiffness matrix
f = zeros(tDof,1); % loading vector

%% Stiffness matrix computation
order = 2;
% [W, Q] = quadrature(order,'GAUSS',2); % 2, standard element
W = [1; 1; 1; 1];
Q = [+1/sqrt(3) +1/sqrt(3);
     +1/sqrt(3) -1/sqrt(3);
     -1/sqrt(3) +1/sqrt(3);
     -1/sqrt(3) -1/sqrt(3)];


for iEle = 1 : numElem
    nEle = element(iEle,:); % element connectivity
    nn   = 4;

    % Stiffness matrix Ke = B^T C B
    
    for k = 1 : nn
        indexK(2*k-1) = 2*nEle(k)-1 ;
        indexK(2*k)   = 2*nEle(k)   ;
    end
    
    ke = zeros(8,8);
    for iGp = 1 : size(W,1)
        pt = Q(iGp,:);                             % quadrature point
        % B matrix
        [N,dNdxi] = lagrange_basis(elemType,pt);
        J0 = node(nEle,:)'*dNdxi;
        invJ0 = inv(J0);
        dNdx  = dNdxi*invJ0;                      % derivatives of N w.r.t XY
        
        Bfem = zeros(3,2*nn);
        Bfem(1,1:2:2*nn)  = dNdx(:,1)' ;
        Bfem(2,2:2:2*nn)  = dNdx(:,2)' ;
        Bfem(3,1:2:2*nn)  = dNdx(:,2)' ;
        Bfem(3,2:2:2*nn)  = dNdx(:,1)' ;
        
        % Stiffness matrix
        ke = ke + Bfem'*C*Bfem*W(iGp)*det(J0);
    end                  % end of looping on GPs
    K(indexK,indexK) = K(indexK,indexK) + ke;
end                      % end of looping on elements

%% -------------------------------------
%  Transform these Gauss points to global coords for plotting
gp = [];
for iel = 1 : numElem
    sctr = element(iel,:);
    for igp = 1 : size(W,1)
        gpnt = Q(igp,:);
        [N,dNdxi]=lagrange_basis('Q4',gpnt);
        Gpnt = N' * node(sctr,:); % global GP
        gp = [gp;Gpnt];
    end
end
figure
hold on
scatter(node(:,1),node(:,2),20);
plot(gp(:,1),gp(:,2),'r+');
axis equal

%% FORCE VECTOR
% disp([num2str(toc),'   NODAL FORCE VECTOR COMPUTATION'])

w1D = [1; 1];
q1D = [1/sqrt(3); -1/sqrt(3)];
% The top edge is applied a traction along Y direction
for iF = 1:(length(topNodes)-1)
    sctr = [topNodes(iF) topNodes(iF+1)];
    indexF = sctr.*2 ;

    for q = 1:size(w1D,1)
        pt = q1D(q,:);
        wt = w1D(q);
        N  = lagrange_basis('L2',pt);
        J0 = abs(node(sctr(2))-node(sctr(1)))/2;
        f(indexF) = f(indexF) + N*(sigmato)*det(J0)*wt;
    end   % of quadrature loop
end       % of element loop


%%  ESSENTIAL BOUNDARY CONDITION

DOFClamped = [1 2 11 12];

Kred = K; 
fred = f;

Kred(DOFClamped,:) = [];
Kred(:,DOFClamped) = [];
fred(DOFClamped,:) = [];

%% SOLUTION OF EQUATIONS
% disp([num2str(toc),'   LINEAR SOLUTION'])
uTemp=Kred\fred;


u = zeros(2*numNode,1);
indexGlobal = [1:2*numNode]';
indexReduce = indexGlobal;
indexReduce(DOFClamped,:) = [];

u(indexReduce) = uTemp;

ux=u(1:2:length(u));
uy=u(2:2:length(u));

%% POST PROCESSING
% Plot numerical deformed configuration
figure
hold on
axis equal
colormap jet
scale = 1000;
for ie = 1:numElem
    iN = element(ie,:);
    fill(node(iN,1),node(iN,2),uy(iN));
end

%% Compute stress at nodes and plot
figure
hold on
axis equal
colormap jet
for ie = 1:numElem
    iN = element(ie,:);
    fill(node(iN,1),node(iN,2),node(iN,1),'EdgeColor', 'none');
end

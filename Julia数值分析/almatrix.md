# 线性方程组

## Gauss消元法

对于线性方程组

$$\left\{\begin{aligned}
a_{11}x_1 +a_{12}x_2 +&\ldots+a_{1n}x_n &= b_1,\\
a_{11}x_1 +a_{12}x_2 +&\ldots+a_{1n}x_n &= b_2,\\
&\vdots\\
a_{n1}x_1 +a_{n2}x_2 +&\ldots+a_{nn}x_n &= b_n,
\end{aligned}\right.
$$

其矩阵形式为$Ax=b$，其中
$$A=\begin{bmatrix}
a_{11}&a_{12}&\ldots&a_{1n}\\ 
a_{21}&a_{22}&\ldots&a_{2n}\\
&&\vdots&\\ 
a_{n1}&a_{n2}&\ldots&a{nn}\\ 
\end{bmatrix},
x=\begin{bmatrix}
x_1\\x_2\\\vdots\\x_n\\
\end{bmatrix},
b=\begin{bmatrix}
b_1\\b_2\\\vdots\\b_n\\
\end{bmatrix}
$$

所谓消去法，即逐次小区未知数的方法，将$Ax=b$化为与原式等价的三角方程组。若化为如下形式

$$\begin{bmatrix}
a_{11}&a_{12}&\ldots&a_{1n}\\ 
0&a_{22}&\ldots&a_{2n}\\
\vdots&\ldots&\vdots&\\ 
0&0&\ldots&a{nn}\\ 
\end{bmatrix}\cdot
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_n\\
\end{bmatrix}
=\begin{bmatrix}
b_1\\b_2\\\vdots\\b_n\\
\end{bmatrix}
$$

则易得求解公式

$$\left\{\begin{aligned}
x_n&=b_n/a_{nn}\\
x_k&=(b_k-\displaystyle\sum_{j=k+1}^na_{kj}x_j)/a_{kk}   
\end{aligned}\right.
$$

在Julia中可写为

```julia
# Gauss消去法
# M为n×n+1矩阵，对应Ax=b,M[:,end]为b
function GaussElimination(M)
    M = copy(M)
    n,_ = size(M)
    # 消元过程
    for i in 1:n-1
        vMinus = M[i,:]/M[i,i]  #当前行循环所要消去的变量
        for j in i+1:n
            M[j,:] -= vMinus*M[j,i]
        end
    end
    # 回代过程
    x = zeros(n)
    for i in n:-1:1
        sumValue = sum([M[i,j]*x[j] for j in i:n])
        x[i] = (M[i,end]-sumValue)/M[i,i]
    end
    return x
end
```

验证

```julia
julia> A = rand(10,10);
julia> x = rand(10);
julia> b = A*x;
julia> x1 = GaussElimination(hcat(A,b));#hcat函数为矩阵拼接
julia> x1-x     
10-element Array{Float64,1}:
  3.3306690738754696e-15
 -1.021405182655144e-14
  5.495603971894525e-15
  5.2735593669694936e-15
 -5.10702591327572e-15
 -8.604228440844963e-15
  5.329070518200751e-15
  1.0380585280245214e-14
 -1.5836670082952642e-14
  8.659739592076221e-15
```
## 矩阵的三角分解

在Gauss消元的过程中，我们构造了一个上三角矩阵用于回代，在这个过程中，每一步操作均类似于矩阵的初等变换，而初等初等变换又可以等效为矩阵乘法，所以这些变换矩阵之间的乘积又构成了另一个矩阵。

即高斯消元过程为

$$
L_n*\ldots*L_2*L_1*M = U
$$

其中，$U$为上三角阵，则$L_1^{-1}*L_2^{-1}*\ldots*L_{n-1}^{-1}*U=M$，令$L=L_1^{-1}*L_2^{-1}*\ldots*L_{n-1}^{-1}$，则有$LU=M$。

通过Julia实现为

```julia
# 生成单位矩阵
function eye(n)
    M = zeros(n,n)
    for i in 1:n
        M[i,i] = 1
    end
    return M
end

# 矩阵的LU分解
function LU(M)
    U = copy(M)
    n,_ = size(M)
    L = eye(n)
    eleMatrix = eye(n)
    # 初等变换过程
    for i in 1:n-1
        eleMatrix[i+1:end,i] = [-U[j,i]/U[i,i] for j in i+1:n]
        U = eleMatrix*U
        L = L/eleMatrix
        eleMatrix[i+1:end,i] = [0 for j in i+1:n]
    end
    return L,U
end
```

验证

```julia
julia> M = rand(4,4)
julia> L,U = LU(M);

julia> L
4×4 Array{Float64,2}:
 1.0        0.0      0.0       0.0
 0.0795445  1.0      0.0       0.0
 0.686459   1.03633  1.0       0.0
 0.327521   1.27914  0.659519  1.0

julia> U
4×4 Array{Float64,2}:
 0.591548  0.102808   0.599782      0.25122
 0.0       0.51307    0.902331      0.481771
 0.0       0.0       -0.691109      0.258695
 0.0       0.0       -5.55112e-17  -0.170634

julia> L*U-M
4×4 Array{Float64,2}:
 0.0  0.0          0.0  0.0
 0.0  0.0          0.0  0.0
 0.0  0.0          0.0  0.0
 0.0  1.11022e-16  0.0  0.0
```
## Gauss 主元素消去法

Gauss消元公式表明，$a_{kk}$项不可为0，否则将出现除零错误。非但如此，如果$a_{kk}$是一个小量，由于计算机程序在运行过程中，其位数有限，所以可能会造成较大误差。

```julia
julia> A =rand(4,4)*10;
julia> A[1,1] = 1e-10;
julia> x = rand(4)*10;
julia> b = A*x;
julia> x1 = GaussElimination(hcat(A,b));

julia> x1-x
4-element Array{Float64,1}:
  0.00014665020004134277
 -0.00015594355145509553
  1.9684886276571945e-6
  1.596867372199995e-5
```

可以看到，其最大误差已经达到了0.0001。

为了避免这种误差，最直观的想法是，在Gauss消元的过程中，使得较大元素的值尽量靠前。对于增广矩阵$B=[A|b]$而言，若在$A$中有最大元素$a_{ij}$，则可以将$B$的第$i$行，第$j$列与第一行、第一列进行交换。对于交换后的矩阵，则选取除了第一行第一列之外的子矩阵继续进行最大值的选取与交换，直至进行到最后一行。

其Julia实现为

```julia
# M为n×n+1增广矩阵，对应Ax=b,M[:,end]为b
# 输出为最大元调整后的矩阵和未知数的变化顺序
function allPrinciple(M)
    M = copy(M)
    n,_ = size(M)
    J = zeros(n-1)        #保存x的交换信息
    for i = 1:n-1
        pos = findmax(M[i:end,i:end-1])[2]
        ni = pos[1]+i-1; nj = pos[2]+i-1    #最大值所在位置
        J[i] = nj
        # 交换行
        rowi = M[i,:]
        M[i,:] = M[ni,:]
        M[ni,:] = rowi
        # 交换列
        colj = M[:,i]
        M[:,i] = M[:,nj]
        M[:,nj] = colj
    end
    return M,J
end
```

验证

```julia
julia> include("almatrix.jl");
julia> A = rand(4,4)*10;
julia> A[1,1] = 1e-10;
julia> x = rand(4)*10;
julia> b = A*x;
julia> M = hcat(A,b)
4×5 Array{Float64,2}:
 1.0e-10  3.15827   6.67401  6.19187  91.509
 1.33812  5.68155   1.95993  9.98016  91.6819
 9.29672  1.16334   1.35398  2.7712   46.9376
 5.32787  0.445123  3.25703  1.22906  41.3079
 julia> M1,J=allPrinciple(M);
 julia> M1
4×5 Array{Float64,2}:
 9.98016  1.33812  1.95993  5.68155   91.6819
 2.7712   9.29672  1.35398  1.16334   46.9376
 6.19187  1.0e-10  6.67401  3.15827   91.509
 1.22906  5.32787  3.25703  0.445123  41.3079
```

可见已经实现了最大元交换，接下来可将换元过程插入到Gauss消去法中，

```julia
#全主元素消去法
function allElimitation(M)
    M,J = allPrinciple(M)       #最大主元替换
    x = GaussElimination(M)     #高斯消去法
    for i in length(J):-1:1     #未知数换位
        print(J[i])
        xi = x[J[i]]
        x[J[i]] = x[i]
        x[i] = xi
    end
    return x
end
```julia

验证(接上面代码)

```julia
julia> include("almatrix.jl");
julia> x1 = GaussElimination(M);
julia> x2 = allElimitation(M);
julia> x1-x
4-element Array{Float64,1}:
 -0.00024191569586218264
 -0.0018563311857411335
 -0.00014663572230411148
  0.0011049087452414952

julia> x2-x
4-element Array{Float64,1}:
  6.661338147750939e-16
  2.1316282072803006e-14
 -8.881784197001252e-16
 -1.1546319456101628e-14
```

可见主元素消去法相对于原始的Gauss消去法提高了不少精度。

## Gauss-Jordan消去法

Gauss消去法是一种从上到下的消元过程，即始终消去对角线下方的元素，最终使得系数矩阵变成上三角的形式。Gauss-Jordan消去法则还要消去对角线上方的元素，使得系数矩阵变成对角矩阵。

```julia
# Gauss-Jordan消去法
# M为n×n+1增广矩阵，对应Ax=b,M[:,end]为b
function GaussJordan(M,maxFlag=true)
    if maxFlag
        M,J = allPrinciple(M)       #最大主元替换
    end

    n,_ = size(M)
    # Gauss消元
    for i in 1:n-1
        vMinus = M[i,:]/M[i,i]      #当前行循环所要消去的变量
        for j in i+1:n
            M[j,:] -= vMinus*M[j,i]
        end
    end

    # Jordan消元
    for i in n:-1:2
        vMinus = M[i,:]/M[i,i]
        for j in 1:i-1
            M[j,:] -= vMinus*M[j,i]
        end
    end

    # 回代
    x = [M[i,end]/M[i,i] for i in 1:n]

    if maxFlag
        for i in length(J):-1:1     #未知数换位
            print(J[i])
            xi = x[J[i]]
            x[J[i]] = x[i]
            x[i] = xi
        end
    end
    return M,x
end
```
## 追赶法

在某些偏微分方程的求解算法中，在初值条件给定的情况下会产生一个特殊的方程求解问题，其系数矩阵为三对角矩阵$Ax=f$，其形式如下

$$\begin{bmatrix}
b_1&c_1&&& \\
a_2&b_2&c_2&&\\
&\ddots&\ddots&\\
&&a_{n-1}&b_{n-1}&c_{n-1}\\
&&&a_n&b_n
\end{bmatrix}
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_{n-1}\\x_n  
\end{bmatrix}=
\begin{bmatrix}
f_1\\f_2\\\vdots\\f_{n-1}\\f_n  
\end{bmatrix}
$$

此时，对A进行$LU$分解可得

$$A=\begin{bmatrix}
b_1&c_1&&& \\
a_2&b_2&c_2&&\\
&\ddots&\ddots&\\
&&a_{n-1}&b_{n-1}&c_{n-1}\\
&&&a_n&b_n
\end{bmatrix}
=\begin{bmatrix}
\alpha_1&&&\\
\gamma_2&\alpha_2&&\\
&\ddots&\ddots\\
&&\gamma_n&\alpha_n
\end{bmatrix}
\begin{bmatrix}
1&\beta_1&& \\
&\ddots&\ddots\\
&&1&\beta_{n-1}\\
&&&1
\end{bmatrix}
$$

易得其变换关系为

$$\left\{\begin{aligned}
b_1&=\alpha_1,\quad c_1=\alpha_1\beta_1,\\
a_i&=\gamma_i,\quad b_i = \gamma_i\beta_{i-1}+\alpha_i \quad i= 2,3,\ldots,n\\
c_i&= \alpha_i\beta_i \quad i=2,3,\ldots,n-1
\end{aligned}\right.
$$

根据上式可得到追赶法的求解过程

Step1 计算$\beta_i$

$$
\beta_1=c_1/b_1,\quad \beta_i=c_i/(b_i-a_i\beta_{i-1})\quad i=2,3,\ldots,n-1;
$$

step2 计算$y$

$$
y1 = f_1/b_1,\quad y_i=(f_i-a_iy_{i-1})/(b_i-a_i\beta_{i-1})\quad i=2,3,\ldots,n
$$

step3 计算$x$，即其追赶过程

$$
x_n=y_n, \quad x_i=y_i-\beta_ix_{i+1},\quad i=n-1,n-2,\ldots,1
$$

其Julia实现为

```julia
# 追赶法,输入M为三对角矩阵，f为向量
function chasing(M,f)
    n,_ = size(M)
    b = [M[i,i] for i in 1:n]
    c = [M[i,i+1] for i in 1:n-1]
    a = vcat(1,[M[i+1,i] for i in 1:n-1])

    # 根据递推公式求beta
    beta = zeros(n-1)
    beta[1] = c[1]/b[1]
    for i in 2:n-1
        beta[i] = c[i]/(b[i]-a[i]*beta[i-1])
    end

    # 求解y
    y = zeros(n)
    y[1] = f[1]/b[1]
    for i in 2:n
        y[i] = (f[i]-a[i]*y[i-1])/(b[i]-a[i]*beta[i-1])
    end

    # x的追赶过程
    x = zeros(n)
    x[n] = y[n]
    for i in n-1:-1:1
        x[i] = y[i]-beta[i]*x[i+1]
    end
    
    return x
end
```

验证

```julia
julia> M = zeros(5,5);
julia> for i in 1:5
       M[i,i] = rand()*10
       M[i,i+1] = rand()*3
       M[i+1,i] = rand()*3
       end          #此处会报错哦
julia> x = rand(5);
julia> f = M*x;
julia> x1 = chasing(M,f);
julia> x1-x
5-element Array{Float64,1}:
  0.0
 -2.220446049250313e-16
  0.0
 -1.1102230246251565e-16
  0.0
```

## Jocabi迭代法

对于方程组$Ax=b$

$$
\displaystyle\sum_{j=1}^na_{ij}x_j=b_j,\quad i=1,2,\ldots,n
$$

若$A$为非奇异矩阵且任意项不为零，则将$A$分裂为$A=D-L-U$,其中$D$为对角矩阵，$L,U$分别为对焦为0的下三角和上三角矩阵。

将方程组中的对角项提出，则方程组变为

$$
x_i=\frac{1}{a_{ii}}(b_i-\displaystyle\sum_{j=1,j\not=i}^na_{ij})
$$

或将矩阵表达式写为

$$(L+U)x-Dx=b \to x=D^{-1}((L+U)x-b)$$

记为

$$x=B_0x+f$$

其中

$$B_0=I-D^{-1}A=D^{-1}(L+U),\quad f=D^{-1}b$$

通过迭代法求上式即可得到$Jacobi$迭代公式

$$\left\{\begin{aligned}
x^{(0)}&=[x_1^{(0)},x_2^{(0)},\ldots,x_n^{(0)}]^T\\
x_i^{(k+1)}&=\frac{1}{a_{ii}}(b_i-\displaystyle\sum_{j=1,j\not=i}^na_{ij}x_j^{(k)})
\end{aligned}\right.
$$

写成矩阵形式即为

$$\left\{\begin{aligned}
x^{(0)}&\\
x^{(k+1)}&=B_0x^{(k)}+f
\end{aligned}\right.
$$

其Julia实现为

```julia
## Jocobi迭代法
## n为迭代次数
function Jacobi(M,b,n=10)
    nDim,_ = size(M)
    D = zeros(nDim,nDim)
    for i in 1:nDim
        D[i,i] = M[i,i]
    end
    B = D\(D-M)
    f = D\b

    x = f       #初始化
    for i = 1:n
        x = B*x+f
    end
    return x
end
```


## Gauss-Seidel迭代

在Jacobi迭代中，$x$被看成一个整体参与迭代，在每一次迭代的过程中，同一代的$x$其内部的变量并无相互影响。

然而，如果迭代收敛，那么新一代的$x$一定比旧一代的$x$更接近真实值，所以，如果将迭代过程拆分，让新一代的计算结果实时取代旧一代的结果，那么应该有助于加速收敛过程，此即Gauss-Seidel迭代。

如果写出迭代公式的展开式可能会更加直观

$$\left\{\begin{aligned}
x_1^{(k+1)}&=\frac{1}{a_{11}}(b_1-\displaystyle\sum_{j=1,j\not=1}^na_{1j}x_j^{(k)})\\
x_2^{(k+1)}&=\frac{1}{a_{22}}(b_1-\displaystyle\sum_{j=1,j\not=2}^na_{2j}x_j^{(k)})\\
\vdots\\
x_n^{(k+1)}&=\frac{1}{a_{nn}}(b_1-\displaystyle\sum_{j=1,j\not=n}^na_{nj}x_j^{(k)})
\end{aligned}\right.
$$

则Gauss-Seidel迭代可写为

$$\left\{\begin{aligned}
x_1^{(k+1)}&=\frac{1}{a_{11}}(b_1-a_{12}x_2^{(k)}-a_{13}x_3^{(k)}-\ldots-a_{1n}x_n^{(k)}\\
x_2^{(k+1)}&=\frac{1}{a_{11}}(b_1-a_{12}x_2^{(k+1)}-a_{13}x_3^{(k)}-\ldots-a_{1n}x_n^{(k)}\\
\vdots\\
x_n^{(k+1)}&=\frac{1}{a_{11}}(b_1-a_{12}x_2^{(k+1)}-a_{13}x_3^{(k+1)}-\ldots-a_{1,n-1}x_{n-1}^{(k+1)}
\end{aligned}\right.
$$

# 矩阵特征值与特征向量

## Gerschgorin定理

设$A=(a_{ij})_{m\times n})$，则$A$的每一个特征值必属于下述圆盘之中

$$
|\lambda-a_{ii}|\leqslant\displaystyle\sum_{j=1,j\not=i}^{n}|a_{ij}|
$$

证明 设$\lambda$为$A$内任意一个特征值，$x$为对应的特征向量，则有

$$(\lambda I-A)x=0$$

记$x=(x_1,x_2,\ldots,x_n)^T\not=0$以及$|x_i|=\max_k|x_k|,x\not=0$,则

$$(\lambda-a_{ii})x_i = \displaystyle\sum_{j=1,j\not=i}^na_{i,j}x_j$$

又因$|xj/xi|\leqslant1$,则

$$|\lambda-a_{ii}| \leqslant\displaystyle\sum_{j=1,j\not=i}^n|a_{i,j}|$$


## 幂法和反幂法

幂法时一种迭代算法，所以其基本表示形式必然为$\nu_{n+1}=A\nu_{n}$，即任取一非零向量$\nu_0$，通过矩阵$A$构造一向量序列

$$\left\{\begin{aligned}
\nu_1&=A\nu_0\\
\nu_2&=A\nu_1=A^2\nu_0\\
&\vdots\\
\nu_{k+1}&=A\nu_k=A^{k+1}\nu_0\\
&\vdots
\end{aligned}\right.
$$

若令$\nu_0=\sum^n_{i=1}a_ix_i,\quad\lambda_i$为相应的特征向量，则

$$\begin{aligned}
\nu_k&=A\nu_{k-1}=\displaystyle\sum^n_{i=1}a_i\lambda_i^kx_i\\
&=\lambda_i^k[a_1x_1+\displaystyle\sum^n_{i=2}a_i(\lambda_i/\lambda_1)^kx_i
\end{aligned}
$$

若令$|\lambda_i|>|\lambda_{i+1}|$，则$|\lambda_i/\lambda_1|<1$，则$\lim\limits_{k\to\infty}(\lambda_i/\lambda_1)^k=0$。即经过无穷多次迭代之后

$$\nu_k\approx a_1\lambda_1^kx_1$$

此即矩阵的最大特征值所对应的特征向量，相邻两次迭代特征向量之商即为最大特征值。这种方法即为$\textbf{幂法}$，与之相应地，由于其逆矩阵的特征值变为原矩阵的倒数，则对逆矩阵使用幂法，即可得到最小特征向量，此方法即为$\textbf{反幂法}$

```julia
# 乘幂法
function Power(M,n)
    arr = rand(size(M)[1])
    for i in 1:n-1
        arr = M*arr
        arr = arr/maximum(arr)  #对向量进行归一化
    end
    Arr = M*arr                 
    return Arr,Arr./arr
end
```

验证

```julia
julia> include("almatrix.jl");
julia> M = rand(5,5);
julia> Power(M,20)[2]
5-element Array{Float64,1}:
 2.6429750659456106         #此即最大特征值
 2.6429750659461253
 2.6429750659454907
 2.6429750659466134
 2.642975065945339
```

由于向量之间点除很难得到相同的值，所以我们选择其中一点来观察乘幂法的收敛速度。

令$i_k$为$\nu_k$的第$i$个分量，$\varepsilon^i_k$为第$k$次迭代误差的第$i$个分量，则

$$
\frac{i_{k+1}}{i_k}=\lambda_1\frac{a_1x_1^i}{a_1x_1^i}+\lambda_2\frac{a_2x_2^i}{a_2x_2^i}+\ldots+\lambda_n\frac{a_1x_n^i}{a_1x_n^i}=
\lambda_1\frac{a_1x_1^i+\varepsilon^i_{k+1}}{a_1x_1^i+\varepsilon^i_k}
$$

根据其误差分量的特征可知，当主特征值$\lambda_1$远大于其他特征值时，将具备更快的收敛速度。若$\lambda_2\approx\lambda_1$，则收敛速度将会很慢。所以，如果能够使得$\lambda_1$与其他特征值的差距变大，则可增快收敛速度。

对此，可以构造矩阵$B=A-pI$，则其特征值为$\lambda_i-p$，则$B$的收敛速度大于$A$，因为

$$
|\frac{\lambda_2-p}{\lambda_1-p}|<|\frac{\lambda_2}{\lambda_1}|
$$

通过Julia写为

```julia
# 加速之后的幂法
function acPower(M,n,p)
    B = M .- eye(size(M)[1])*p
    arr,val = Power(B,n)
    return arr,val.+p
end
```

测试

```julia
julia> x = rand(5,5);
julia> x[5,5] = 7;
julia> x[4,4] = 8;
julia> x[3,3] = 8.5;
julia> x[2,2] = 9.5;
julia> x[1,1] = 10;
julia> acPower(x,20,0)[2]
5-element Array{Float64,1}:
 11.119295564775351
 11.079539551635982
 11.058127553729367
 11.095918606361074
 11.05404344237109
 julia> acPower(x,15,5)[2]
5-element Array{Float64,1}:
 11.090309381320136
 11.094079685109394
 11.09547046795793
 11.094168499705209
 11.094043676186722
```

可见，在p值选5的情况下，迭代15次的结果优于未加速时迭代20次的结果。（因为这五个值都是主特征向量，所以这五个值越相近说明越接近真实值。）


## Schmidt正交分解

设$\alpha_i$是$R^n$上的$n$个线性无关的向量，令$\beta_1=\frac{\alpha_1}{\|\alpha_1\|_2}$，则$\beta_1$为单位向量，再令$B_2=\alpha_2-(\alpha_2,\beta_1)\beta_1 \to \beta_2=\frac{B2}{\|B_2\|_2}$，则$(\beta_1,\beta_2)=0$。

同理，得到对任意向量组的正交化过程

$$\begin{aligned}
\alpha_1&=\alpha_1,\quad &\beta_1&=\frac{\alpha_1}{\|\alpha_1\|_2}\\
B_2&=\alpha_2-(\alpha_2,\beta_1)\beta_1\quad&\beta_2&=\frac{B2}{\|B_2\|_2}\\
B_3&=\alpha_3-(\alpha_3,\beta_1)\beta_1-(\alpha_3,\beta_2)\beta_2\ &\beta_3&=\frac{B_3}{\|B_3\|_2}\\
&\ldots&&\ldots\\
B_n&=\alpha_n-\displaystyle\sum_{j=1}^{n-1}
(\alpha_n,\beta_j)\beta_j&\beta_n&=\frac{B_n}{\|B_n\|_2}\\
\end{aligned}
$$

这个过程可以表示为矩阵形式

$$\begin{aligned}
[\alpha_1,\ldots,\alpha_n]=&[\beta_1,\ldots,\beta_n]
\begin{bmatrix}
\|\beta_1\|_2&(\alpha_2,\beta_1)&(\alpha_3,\beta_1)&\ldots&(\alpha_n,\beta_1)\\
&\|\beta_2\|_2&(\alpha_3,\beta_2)&\ldots&(\alpha_n,\beta_1)\\
&&\|\beta_3\|_2&\ldots&(\alpha_n,\beta_1)\\
&&&\ldots&\vdots\\
&&&&\|\beta_n\|_2
\end{bmatrix}\\
&=QR
\end{aligned}
$$

其中，Q为正交阵，R为上三角阵，上三角阵的对角元素即为其特征值。其Julia实现为

```julia
# Schmidt正交化法，输入为矩阵,按每一列求其归一化向量
function Schmidt(M)
    n,_ = size(M)
    N = zeros(n,n)
    for i = 1:n
        B = M[:,i]
        for j in 1:i-1
            B -= M[:,i]'*N[:,j]*N[:,j]
        end
        N[:,i] = B/sqrt(sum([B[j]^2 for j in 1:n]))
    end
    return N
end
```

验证

```julia
julia> include("almatrix.jl");
julia> x = rand(5,5);
julia> y = Schmidt(x);
julia> for i in 1:5
       println(y[:,1]'*y[:,i])
       end
1.0000000000000002
-1.1102230246251565e-16
-3.3306690738754696e-16
-9.645062526431047e-16
-8.326672684688674e-16
```

## Jacobi旋转法

Jacobi法的基本思想是构造简单、特殊的正交矩阵序列$\{P_k\}$，使得

$$\begin{aligned}
A_0=&A\\
A_{k+1}=&P_kA_kP_k^T,\quad k=0,1,2,\ldots
\end{aligned}
$$

逐渐接近对角型，从而得到A的特征值。

对于矩阵

$$
P_{ij}=\begin{bmatrix}
1\\
&\ddots\\
&&\cos{\theta}&\ldots&\sin{\theta}\\
&&\vdots&1&\vdots\\
&&-\sin{\theta}&\ldots&\cos{\theta}\\
&&&&&\ddots\\
&&&&&&1
\end{bmatrix}
$$

其中，对角上的$\cos{\theta}，\sin{\theta}$分别在第$i,j$列，则该矩阵被称为平面旋转矩阵。

设矩阵$A\in R^{m\times n}$是对称矩阵，记$A+0=A$，对$A$做一系列旋转相似变换，得$C=P^TAP$,则$A、C$只在第$i,j$行列的元素不同，且有如下关系

$$\left\{\begin{aligned}
c_{ii} =& a_{ii}\cos^2\theta+a_{jj}\sin^2\theta-a_{ij}\sin2\theta\\
c_{jj} =& a_{ii}\sin^2\theta+a_{jj}\cos^2\theta+a_{ij}\sin2\theta\\
c_{ij} =& \frac{1}{2}(a_{ii}-a_{jj})\sin2\theta+a_{ij}\cos2\theta\\
c_{ik} =& c_{ki} =a_{ik}\cos\theta-a_{jk}\sin\theta\\
c_{jk} =& c_{kj} =a_{jk}\cos\theta+a_{ik}\sin\theta\\
\end{aligned}\right.
$$

可以看到，我们可可以通过适当选取$\theta$值，使得$c_{ij}=0$，通过不断选取不同的$\theta$可以逐渐使得矩阵中越来越多的元素为零。

此时，需要满足

$$\begin{aligned}
\tan2\theta=&\frac{-2a_{ij}}{a_{ii}-a_{jj}}\\
\to\cos\theta=&\frac{1}{\sqrt{1+\tan^2\theta}}\\
\to\sin\theta=&\frac{\tan\theta}{\sqrt{1+\tan^2\theta}}\\
\end{aligned}$$

```julia
# Jacobi旋转法
function JacobiRot(M)
    n,_=size(M)
    E = eye(n)
    for i = 1:n
        for j = 1:n
            p2Tan = -2*M[i,j]/M[i,i]-M[j,j]
            p2Cos = 1/sqrt(1+p2Tan^2)
            p2Sin = p2Tan*p2Cos
            pCos = sqrt(p2Cos+1)/2
            pSin = p2Sin/pCos/2
            
            cii = M[i,i]*pCos^2+M[j,j]*pSin^2-M[i,j]*p2Sin
            cjj = cii+2*M[i,j]*p2Sin
            cij = (M[i,i]-M[j,j])*p2Sin/2+M[i,j]*p2Cos
            ci = [M[i,k]*pCos-M[j,k]*pSin for k in 1:n]
            cj = [M[j,k]*pCos+M[i,k]*pSin for k in 1:n]
            ci[i] = cii;ci[j]=cij
            cj[i] = cij;cj[j]=cjj

            M[i,:] = ci;M[:,i] = ci
            M[j,:] = cj;M[:,j] = cj
        end
    end
    return M
end
```

此外还满足如下关系

$$\begin{aligned}
c_{lk} &= c_{lk}\\
c^2_{il}+c^2_{jl}&=a^2_{il}+a^2_{jl}\\
c^2_{ii}+c^2_{jj}+2c^2_{ij}&=a^2_{ii}+a^2_{jj}+2a^2_{ij}\\
\end{aligned}
$$

$\quad所有等式中:l,k\not=i,j$,引入记号

$$\sigma(A)=\displaystyle\sum^n_{i=j}a^2_{ij},\quad\tau(A)=\displaystyle\sum^n_{i\not=j}a^2_{ij}$$

则有

$$\sigma(C)=\sigma(A)+2a^2_{ij},\quad\tau(C)=\tau(A)-2a^2_{ij}$$

这说明每次Jacobi迭代虽然会使得某个元素变为0，但也可能会让某些为0的元素变为非0，在使用Jacobi旋转法时，也应该遵循最大元优先的原则。

# 插值法

设函数$y=f(x)$在区间$[a,b]$上有定义，且在点$a\leqslant x_0\leqslant x_1\ldots\leqslant x_n\leqslant b$上的值$y_0, y_1 \ldots y_n$之间存在一个函数$P(x)$,使得

$$P(x_i)=y_i (i=0,1,\ldots, n)$$

成立，则称$P(x)$为$f(x)$的插值函数，在$x_0, x_1,\ldots,x_n$称为插值节点，包含节点的区间称为插值区间。如果$P(x)$是次数不超过$n$的代数多项式，则称之为插值多项式。

## Lagrange插值

### 数学原理

若我们有$n+1$组一一对应的$(x,y)$点对，假设存在一多项式$L_n(x)$，使之满足$L_n(x_j)=y_j,j=0,1,\ldots,n$，即

$$\left\{\begin{aligned}
a_0+a_1x_0+\ldots+a_nx_0^n&=&y_0\\
a_0+a_1x_1+\ldots+a_nx_1^n&=&y_1\\
&\vdots&\\
a_0+a_1x_n+\ldots+a_nx_n^n&=&y_n\\
\end{aligned}\right.
$$

则$L_n$即为该组数据的插值多项式。对于任意$j=0,1,\ldots,n$，有$L_n(x_j)=y_j$,则可令

$$
L_n(x)=\displaystyle\sum_{k=0}y_kl_k(x)
$$

其中，

$$
l_k(x_j)=\delta_{jk}=\left\{\begin{aligned}
1, k=j\\
0, k\not=j
\end{aligned}\right.
(j,k=0,1,\ldots,n)
$$

则可满足$L_n(x_j)=\displaystyle\sum_{k=0}^ny_kl_k(x_j)=y_j$，其中$l_k$被称为基函数。根据其所满足的要求，易得基函数的表达式为

$$
l_k(x)=\frac{(x-x_0)\ldots(x-x_{k-1})(x-x_{k+1})\ldots(x-x_n)}{(x_k-x_0)\ldots(x_k-x_{k-1})(x_k-x_{k+1})\ldots(x_k-x_n)}
$$

自此，我们就得到了Lagrange插值多项式。该式简洁明了，其插值基函数规律明显，能使人过目不忘。若记

$$\omega_{n+1}(x)=(x-x_0)(x-x_1)\ldots(x-x_n)$$

则

$$\omega'_{n+1}(x_k)=(x_k-x_0)\ldots(x_k-x_{k-1})(x_k-x_{k+1})\ldots(x_k-x_n)$$

那么Lagrange插值公式可写为

$$
L_n(x)=\displaystyle\sum_{k=0}^ny_k\frac{\omega_{n+1}(x)}{(x-x_k)\omega'_{n+1}(x_k)}
$$

特别地，对于两点而言，Lagrange插值退化为线性插值，即

$$
L_2(x)=y_1\frac{x-x_2}{x_1-x_2}+y_2\frac{x-x_1}{x_2-x_1}
$$

### 算法化

在Lagrange插值公式中，$\omega_{n+1}(x)$为多项式乘法。Julia中虽然对函数式编程有着良好的支持，但如果将函数本身作为输出，则函数实现细节将被隐藏，因此我们并不能得到真正的拟合参数。所以我们需要对多项式进行抽象，即将其简化为数组的形式，其表现形式为`[a_0,a_1,...,a_n]`,代表$a_0+a_1x+\ldots+a_nx^n$。

那么多项式的乘法可以通过进位的方式实现。对于常数项而言，并不会改变多项式的幂数；对于一阶项来说，乘以多项式数组，则将该数组向高位移一位。其实现方法为

```julia
# 多项式乘法，形式为an*x^n，建议a[1]为最低位常数项
function polyMulti(a,b)
    na = length(a); nb = length(b)
    c = zeros(na+nb-1)
    for i = 1:na
        c[i:i+nb-1] += a[i]*b
    end
    return c
end
```

在Lagrange插值公式中亦有多项式除法，其实现方法与乘法类似，只不过其运算顺序需要从高位开始，并且被除多项式的阶数要不小于除数多项式。
在此需要注意，和许多编程语言类似，Julia中的对象采用地址引用的形式，在传参之后如果直接改变数组内容，将穿透作用域，使得函数之外的数组也发生变化，因此需要对其进行深拷贝。

```julia
function polyDiv(a,b)
    a = copy(a)         #此为余数
    na = length(a); nb = length(b)
    nc = na-nb+1
    c = zeros(nc)
    for i = 0:nc-1
        c[nc-i] = a[end-i] ÷ b[end]
        a[nc-i:end-i] += -c[nc-i]*b[1:end]
    end
    return c,a
end
```

此外，考虑到Lagrange插值公式中包含连乘运算。其中，分子中的连乘为多项式连乘，分母中的多项式为值连乘，所以再建立连乘函数。

```julia
# 连乘函数，输入为数组，输出为数组中所有数的乘积
function sumMulti(x)
    if length(x) == 1
        return x
    else
        return x[1]*sumMulti(x[2:end])
    end
end
```

Lagrange插值公式中的多项式连乘形式单一，相比之下更像是多项式的两种表示形式的转换，所以再建立针对Lagrange插值公式的多项式转化函数

```julia
#将(x-x1)(x-x2)...(x-xn)化成an*x^n形式,A[1]为最低位(常数项)
#输入为[x1,x2,...,xn]输出为[a1,a2,...,an]
function polySimplify(x)
    if length(x) == 1
        return [-x[1],1]
    else
        return polyMulti([-x[1],1],polySimplify(x[2:end]))
    end
end
```

至此，可以创建Lagrange插值函数。

```julia
# Language插值函数
# x y: 维度相同的列变量
# A: 多项式系数，A[1]为最低位
using LinearAlgebra         #建立单位矩阵需要使用线性代数包
function Lagrange(x,y)
    n = length(x)
    xMat = x .- x' + Matrix{Float64}(I,n,n)
    polyMat = polySimplify(x)
    A = zeros(n)
    for i = 1:n
        quot = polyDiv(polyMat,[-x[i],1])[1]
        A += y[i] * quot ./ sumMulti(xMat[i,:])
    end 
    return A
end
```

函数中的`xMat`为矩阵

$$\begin{bmatrix}
1 & x_1-x_0 & \ldots & x_n-x_0\\
x_0-x_1 & 1 & \ldots & x_n-x_1\\
& & \vdots & \\
x_0-x_n & x_1-x_n & \ldots & 0\\
\end{bmatrix}
$$

`polyMat`为多项式$\omega_{n+1}(x)$。

在命令行中调用得

```julia
julia> include("interpolation.jl")
Lagrange (generic function with 1 method)

julia> x = [i for i = 1:9];

julia> y = x.^8 + x.^6 + x.^2 .+ 1;

julia> Lagrange(x,y)
9-element Array{Float64,1}:
  1.0
  4.470348358154297e-8
  0.9999999850988388
 -1.4901161193847656e-8
 -7.450580596923828e-9
  2.7939677238464355e-9
  1.0
  0.0
  1.0

```

结果表明，其8次项、6次项、2次项和0次项分别为1或者接近于1，其他项分别为0或者接近于0，可见拟合结果尚可。通过`Plots`包对原始数据和拟合结果进行比较

```julia
julia> using Plots
julia> y1 = y;
julia> y2 = [sum([a[i]*x[j]^(i-1) for i=1:9]) for j = 1:9];
julia> plot(x,[y1,y2],st=[:scatter :line])
julia> savefig("Lagrange.png")
```
![Lagrange](img\Lagrange.png)

可见拟合结果与原始数据几乎一致。

## Newton插值

### 差商
对于一列$x_0,x_1,\ldots,x_n$，若与$y_0,y_1,\ldots,y_n$通过映射$f$一一对应，则定义比值

$$f[x_0,x_1]=\frac{f(x_1)-f(x_0)}{x_1-x_0}$$

为$f(x)$关于节点$x_0,x_1$的一阶差商。

类似地，

$$f[x_0,x_1,x_2]=\frac{f[x_0,x_2]-f[x_0,x_1]}{x_2-x_1}$$

为二阶差商，

$$f[x_0,x_1,\ldots,x_k]=\frac{f[x_0,x_1,\ldots,x_{k-2},x_k]-f[x_0,x_1,\ldots,x_{k-1}]}{x_k-x_{k-1}}$$

为$f(x)$关于节点$x_0,x_1,\ldots,x_k$的$k$阶差商。

$k$阶差商可表示为函数值$f(x_0),f(x_1),\ldots,f(x_n)$的线性组合，即

$$f[x_0,x_1,\ldots,x_k]=\displaystyle\sum_{j=0}^k\frac{f(x_j)}{(x_j-x_0)\ldots(x_j-x_{j-1})(x_j-x_{j+1})\ldots(x_j-x_k)}
$$

则

$$
f[x_0,x_1,\ldots,x_k]=\frac{f[x_1,x_2,\ldots,x_k]-f[x_0,x_1,\ldots,x_{k-1}]}{x_k-x_0}
$$

则可通过递归实现两个数组的差商

```julia
# 差商算法,x,y为同等长度的数组
function diffQuotient(x,y)
    if length(x) == 2
        return (y[2]-y[1])/(x[2]-x[1])
    elseif length(x) == 1
        return y[1]
    else
        return (diffQuotient(x[2:end],y[2:end])
               -diffQuotient(x[1:end-1],y[1:end-1]))/(x[end]-x[1])
    end
end
```

### Newton插值

根据差商定义，可得

$$\begin{aligned}
f(x) &= f(x_0) + f[x,x_0](x-x_0)\\
f[x,x_0] &= f[x_0,x_1] + f[x,x_0,x_1](x-x_1)\\
&\vdots\\
f[x,x_0,\ldots,x_{n-1}] &= f[x_0,x_1,\ldots,x_n] + f[x,x_0,x_1,\ldots,x_n](x-x_n)
\end{aligned}
$$

即

$$\begin{aligned}
f(x) =& f(x_0) + f[x_0,x_1](x-x_0)+ f[x_0,x_1,x_2](x-x_0)(x-x_1)\\
&+\ldots+f[x_0,x_1,\ldots,x_n](x-x_0)(x-x_1)\ldots(x-x_{n-1})\\
&+f[x_0,x_1,\ldots,x_n]\omega_{n+1}(x)\\
=&N_n(x)+R_n(x)    
\end{aligned}
$$

其中，

$$\begin{aligned}
N_n(x) =& f(x_0) + f[x_0,x_1](x-x_0)+ f[x_0,x_1,x_2](x-x_0)(x-x_1)+\\
&\ldots+f[x_0,x_1,\ldots,x_n](x-x_0)(x-x_1)\ldots(x-x_{n-1})\\
R_n(x)=&f[x_0,x_1,\ldots,x_n]\omega_{n+1}(x)
\end{aligned}
$$

$N_n(x)$即为牛顿插值公式，且

$$
N_{n+1}(x) = N_{n}(x) + f[x_0,x_1,\ldots,x_{n+1}]\omega_{n+1}(x)
$$

可快速写为Julia

```julia
# Newton插值函数
function Newton(x,y)
    n = length(x)
    A = zeros(n)
    A[1] = y[1]
    for i = 2:n
        p = DiffQuotient(x[1:i],y[1:i])*polySimplify(x[1:i-1])
        A[1:length(p)] += p
    end
    return A
end
```

当然，无论差商还是连乘运算，都会在运算的过程中产生大量的重复，上述代码中并未有效利用已经计算出的值，所以性能上并不占优。因此，我们对Newton插值进行更改

```julia
# Newton插值函数
function Newton(x,y)
    n = length(x)
    A = zeros(n)
    hisQuot = y[1]      #保存差商的值
    hisPoly = [1]
    A[1] = y[1]
    for i = 2:n
        hisQuot = diffQuotient([x[1],x[i]],[hisQuot,diffQuotient(x[2:i],y[2:i])])
        hisPoly = polyMulti(hisPoly,[-x[i-1],1])
        A[1:length(hisPoly)] += hisQuot*hisPoly
    end
    return A
end
```

调用并验证

```julia
julia> include("interpolation.jl");
julia> x = [i for i = 1:9];
julia> y = x.^8 + x.^6 + x.^2 .+ 1;

julia> Newton(x,y)
9-element Array{Float64,1}:
 1.0
 0.0
 1.0
 0.0
 0.0
 0.0
 1.0
 0.0
 1.0
```

## Hermite插值

无论是Newton插值还是Lagrange插值，都只能在数值本身上满足插值函数与数据节点的重合，Hermite插值则要求其导数值相等。

设在节点$a\leqslant x_0\leqslant x_1 \leqslant\ldots\leqslant x_n\leqslant b$上，$y_j=f(x_j),m_j=f'(x_j),j=0,1,\ldots,n$，则插值多项式的条件为

$$
H(x_j)=y_j,H'(x_j)=m_j,j=0,1,\ldots,n
$$

总共有$2n+2$个等式，可唯一确定一个次数不大于$2n+1$的多项式$H_{2n+1}(x)$，设其表达式可通过基函数$\alpha_j,\beta_j$写成

$$
H_{2n+1}(x)=\displaystyle\sum_{j=0}^n[y_j\alpha_j+m_j\beta_j]
$$

则$\alpha_j,\beta_j$需要满足

$$\left\{\begin{aligned}
\alpha_j(x_k) &= \delta_{jk}\\
\alpha_j'(x_k)&= 0\\
\beta_j(x_k) &= 0\\
\beta_j'(x_k) & = \delta_{jk}
\end{aligned}\right.
$$

两个基函数的条件与Lagrange插值中的基函数相似，由于我们的目标函数为$2n+1$次，所以假设

$$\left\{\begin{aligned}
\alpha_j =  (ax+b)l_j^2(x)\\
\beta_j =  (cx+d)l_j^2(x)
\end{aligned}\right.$$

结合二者所满足的基函数条件，可得

$$\left\{\begin{aligned}
\alpha_j(x_j)&=(ax_j+b)l_j^2(x_j)=1\\
\alpha_j'(x_j)&=l_j(x_j)[al_j(x_j)+2(ax_j+b)l_j'(x_j)]=0\\
\beta_j(x_j)&=(cx_j+d)l_j^2(x_j)=0\\
\beta_j'(x_j)&=l_j(x_j)[cl_j(x_j)+2(cx_j+d)l_j'(x_j)]=1
\end{aligned}\right.
$$

解得

$$\left\{\begin{aligned}
\alpha_j &=  [1-2（x-x_j)\displaystyle\sum_{k=0,k\not=j}^n\frac{1}{x_j-x_k}]l_j^2(x)\\
\beta_j &=  (x-x_j)l_j^2(x)
\end{aligned}\right.$$

令$s_j=\displaystyle\sum_{k=0,k\not=j}\frac{1}{x_j-x_k}$,则Hermite插值公式可写为

$$
H_{2n+1}(x)=\displaystyle\sum_{j=0}^n\{
    [y_j+2s_jy_jx_j-x_jm_j]\cdot l_j^2(x)
    +[-2s_jy_j+m_j]\cdot xl_j^2(x)
    \}
$$

其代码为

```julia
# Hermite插值函数
# x,y分别为自变量、因变量，m为导数
function Hermite(x,y,m)
    n = length(x)
    unitMat = Matrix{Float64}(I,n,n)
    xMat = x .- x' + unitMat
    xInverse = 1./xMat - unitMat
    A = zeros(n*2+1)
    for i = 1:n
        quot = polyDiv(polyMat,[-x[i],1])[1]
        lag = quot ./ sumMulti(xMat[i,:])  #Lagrange基函数
        lag = polyMulti(lag,lag)
        sumInverse = sum(xInverse[i,:])
        a0 = (1+2*x[i]*sumInverse)*y[i]-x[i]*m[i]
        a1 = -2*y[i]+m[i]
        α = 1-2
        A += polyMulti(lag,[a0,a1])
    end
    return A
end
```

验证

```julia
julia> include("interpolation.jl");
julia> x = [i for i = 1:9];
julia> y = x.^8 + x.^6 + x.^2 .+ 1;
julia> m = [rand() for i = 1:9] 
julia> a = Hermite(x,y,m);

julia> y1 = y
julia> y2 = [sum([a[i]*x[j]^(i-1) for i=1:18]) for j = 1:9]
julia> plot(x,[y1,y2],st=[:scatter :line])
julia> savefig("hermite.png")
```

![Hermite](img\Hermite.png)

当然，实际上由于我们引入了一组随即变量作为没一点的导数，我们所拟合出的多项式函数自然与输入的多项式不同

```julia
julia> f(x) = sum([a[i]*x^(i-1) for i=1:18])
julia> plot(f,xlims=(1,9))
julia> savefig("realfig.png")
```

![realfig](img\realfig.png)

## 样条插值

### 高阶多项式拟合误差
由于在Hermite多项式拟合的过程中，在采用多项式函数生成$y$后，简单地通过随机值的方式生成了节点处的导数值，所以很难评价其拟合结果。

现我们通过多项式对函数$f(x)=\frac{1}{1+x^2}$进行拟合

```julia
julia> x = [i for i = -5:5];
julia> f(x) = 1/(1+x^2);
julia> df(x) = -2*x/(1+x^2)^2       #f(x)的导函数
julia> y = f.(x);                   #由于x为向量，所以函数括号前加.
julia> m = df.(x);
julia> a = Hermite(x,y,m)           #Hermite插值
julia> fHermite(x) = sum([a[i]*x^(i-1) for i=1:22])
julia> plot([f,fHermite])           #绘制函数f和fHermite
julia> savefig("errFit.png")        #保存图片
```

![errFit](img\errFit.png)

如图可知，高阶多项式函数在对某些数据进行插值的过程中可能会在局部拟合完美，而在另外的部分具有较大误差，这是Runge发现的，所以被称为Runge现象。如果能够在不同的位置通过分段的形式对数据进行拟合，或许能够克服这一问题。

### 三弯矩法

若函数$S(x)$在$[a,b]$区间内二阶可导，且在每个小区间$[x_k,x_{k+1}$上是三次多项式，其中$a\leqslant x_0\leqslant x_1 \leqslant\ldots\leqslant x_n\leqslant b$为给定节点，则称$S(x)$是节点上的三次样条函数，若在节点$x_j$上给定函数值$y_j=f(x_j),j=0,1,\ldots,n$，且$S(x_j)=y_j$，则称$S(x)$为三次样条插值函数。

对于任意一个小区间来说，三次插值需要有四个待定系数，除了端点处所对应的函数值$S_j(x_{j-1})=y_{j-1},S_j(x_j)=y_j$之外，还需要有两个等式共同构成边界条件。常见的边界条件有三种：

1. 已知两端的一阶导数
2. 已知两端的二阶导数
3. 周期边界条件

现假设我们已知任意点的二阶导数，记$S''(x_j)=M_j,h_j=x_j-x_{j-1},j=0,1,\ldots,n$，则$S''(x)$在区间$[x_{j-1},x_j]$之间是$x$的线性函数，且有$S''_j(x_{j-1})=M_{j-1},S''_j(x_j)=M_j$，通过线性插值，可将其表示为

$$
S''_j(x)=\frac{x_j-x}{h_j}M_{j-1}+\frac{x-x_{j-1}}{h_j}M_j
$$

对其积分可得

$$
S_j(x) = \frac{(x_j-x)^3}{6h_j}M_{j-1}+\frac{(x-x_{j-1})^3}{6h_j}M_j+C_1x+C_2
$$

将$S_j(x_{j-1})=y_{j-1},S_j(x_j)=y_j$代入，整理可得

$$
S_j(x) = \frac{(x_j-x)^3}{6h_j}M_{j-1}+\frac{(x-x_{j-1})^3}{6h_j}M_j+(y_{j-1}-\frac{M_{j-1}h_j^2}{6})\frac{x_j-x}{h_j}+(y_j-\frac{M_jh_j^2}{6})\frac{x-x_{j-1}}{h_j}
$$

这个表达式是天然的差分形式，可以直接写出相应的代码，在此不再列出。一般来说，任意一点的二阶导数对于已知数据而言仍旧是一个未知量，需要我们通过分段函数的连续特性进行求解，即要求$S'(x_j-0)=S'(x_j+0)$,求导可得

$$
S'_j(x)=-\frac{(x_j-x)^2}{2h_j}M_{j-1}+\frac{(x-x_{j-1})^2}{2h_j}M_j+\frac{y_j-y_{j-1}}{h_j}-\frac{(M_j-M_{j-1})h_j}{6}
$$

其左右端点为

$$\left\{\begin{aligned}
S'_j(x_j-0)&=\frac{h_j}{2}M_j+\frac{y_j-y_{j-1}}{h_j}-\frac{(M_j-M_{j-1})h_j}{6}\\
S'_j(x_{j-1}+0)&=-\frac{h_j}{2}M_{j-1}+\frac{y_j-y_{j-1}}{h_j}-\frac{(M_j-M_{j-1})h_j}{6}
\end{aligned}\right.
$$

$$\to\left\{\begin{aligned}
S'_j(x_j-0)&=\frac{h_jM_{j-1}}{6}+\frac{h_j}{3}M_j+\frac{y_j-y_{j-1}}{h_j}\\
S'_{j+1}(x_j+0)&=-\frac{h_{j+1}M_j}{3}-\frac{h_{j+1}}{6}M_{j+1}+\frac{y_{j+1}-y_j}{h_{j+1}}\\
\end{aligned}\right.
$$

根据连续性条件，上面两个等式应该处处相等，则该等式可整理为

$$
\mu_jM_{j-1}+2M_j+\lambda_jM_{j+1}=d_j, \quad j=1,2,\ldots,n-1
$$

各参数分别为

$$\left\{\begin{aligned}
\lambda_j &=\frac{h_{j+1}}{h_j+h_{j+1}}\\
\mu_j &=1-\lambda_j=\frac{h_j}{h_j+h_{j+1}}\\
d_j &=6f[x{j-1,x_j,x_{j+1}}]
\end{aligned}\right.
$$

由于在力学上$M$被解释为梁在截面上的弯矩，所以上式被称为三弯矩方程。三弯矩方程再加上此前提到的两个边界条件，即可求出$M$所对应的唯一解。

#### 第一类边界条件

例如，若给定第一类边界条件，即$S'(x_0)=y_0',S'(x_n)=y_n'$，则可得

$$\left\{\begin{aligned}
S'_n(x_n)&=\frac{h_nM_{n-1}}{6}+\frac{h_n}{3}M_n+\frac{y_n-y_{n-1}}{h_n}=y_n'\\
S'_1(x_0)&=-\frac{h_1M_0}{3}-\frac{h_1}{6}M_1+\frac{y_1-y_0}{h_1}=y_0'\\
\end{aligned}\right.
$$

$$\to\left\{\begin{aligned}
M_{n-1}+2M_n&=\frac{6}{h_n}(y_n'-\frac{y_n-y_{n-1}}{h_n})&\to d_n\\
2M_0+M_1&=\frac{6}{h_1}(-y_0'+\frac{y_1-y_0}{h_1})&\to d_0
\end{aligned}\right.
$$

将边界条件与三弯矩方程联立在一起，可得到确定$n+1$个参数的方程组，写成矩阵形式为

$$\begin{bmatrix}
2&1&&&&\\
\mu_1&2&\lambda_1&&&\\
&\mu_2&2&\lambda_2&&\\
&&\ddots&\ddots&\\
&&&\mu_{n-1}&2&\lambda_{n-1}\\
&&&&1&2
\end{bmatrix}
\begin{bmatrix}
M_0\\M_1\\\vdots\\M_{n-1}\\M_{n}  
\end{bmatrix}=
\begin{bmatrix}
d_0\\d_1\\\vdots\\d_{n-1}\\d_{n}  
\end{bmatrix}
$$

#### 第二类边界条件

第二类边界条件为$S''(x_0)=y_0'',S''(x_n)=y_n''$，其矩阵方程可表示为

$$\begin{bmatrix}
2&\lambda_1&&&\\
\mu_2&2&\lambda_2&&\\
&\ddots&\ddots&\\
&&\mu_{n-2}&2&\lambda_{n-2}\\
&&&\mu_{n-1}&2
\end{bmatrix}
\begin{bmatrix}
M_1\\M_2\\\vdots\\M_{n-2}\\M_{n-1}  
\end{bmatrix}=
\begin{bmatrix}
d_1-\mu_1y_0''\\d_2\\\vdots\\d_{n-2}\\d_{n-1}-\lambda_{n-1}y_n''  
\end{bmatrix}
$$

#### 第三类边界条件

第三类边界条件为周期条件，

$$S''(x_0+0)=S''(x_n-0),S'(x_0+0)=S'(x_n-0)$$

$$\to M_0=M_n,\quad \lambda_nM_1+\mu_nM_{n-1}+2M_n=d_n$$

其中，

$$\lambda_n=\frac{h_1}{h_1+h_n},\mu_n=1-\lambda_n=\frac{h_n}{h_1+h_n}$$

$$d_n=\frac{6}{h_1+h_2}(\frac{y_1-y_0}{h_1}-\frac{y_n-y_{n-1}}{h_n})$$

其矩阵方程为

$$\begin{bmatrix}
2&\lambda_1&&&\mu_1 \\
\mu_2&2&\lambda_2&&\\
&\ddots&\ddots&\\
&&\mu_{n-1}&2&\lambda_{n-1}\\
\lambda_n&&&\mu_n&2
\end{bmatrix}
\begin{bmatrix}
M_1\\M_2\\\vdots\\M_{n-1}\\M_n  
\end{bmatrix}=
\begin{bmatrix}
d_1\\d_2\\\vdots\\d_{n-1}\\d_n  
\end{bmatrix}
$$

求解过程代码如下

```julia
# 得到三弯矩法中的弯矩值
# flag表示三类边界条件，需要注意，二类为周期条件，三类为
# para为三类边界条件输入的边界参数
function getMoment(x,y,flag,para=[0,0])
    n = length(x)
    h = x[2:end]-x[1:end-1]
    λ = h[2:end] ./ (h[1:end-1]+h[1:end-1])
    μ = 1 .- λ
    d = [6 * diffQuotient(x[i:i+2],y[i:i+2]) 
            for i in 1:n-2]
    
    nMatrix = Int(n-(-1.5*flag^2+6.5*flag-5))
    M = Matrix{Float64}(I,nMatrix,nMatrix)*2

    # 确定d,λ,μ的值
    if flag == 1
        d0 = 6/h[1]*((y[2]-y[1])/h[1]-para[1])
        dn = 6/h[end]*(-(y[end]-y[end-1])/h[end]+para[2])
        d = vcat(d0,d,dn)
        λ = vcat(1,λ)
        μ = vcat(μ,1)
    elseif flag == 2
        d[1] -= μ[1]*para[1]
        d[end] -= λ[end]*para[2]
        μ = μ[2:end]
    else
        dn = 6/(h[1]+h[end])*((y[2]-y[1])/h[1]-(y[end]-y[end-1])/h[end])
        d = vcat(d,dn)
        M[1,end] = μ[1]
        M[end,1] = λ[end]
        λ = vcat(λ,h[1]/(h[1]+h[end]))
        μ = vcat(μ[2:end],h[end]/(h[1]+h[end]))

    end

    # 生成M矩阵
    for i in 1:nMatrix-1
        M[i,i+1] = λ[i]
        M[i+1,i] = μ[i]
    end

    return M,d
end
```

## 参考资料：

1. 李庆杨.数值分析第四版
2. 颜庆津.数值分析第三版
3. Kincaid.数值分析第三版

# 数值微积分

## 数值微分

导数的定义为

$$f'(a)=\lim\limits_{h\to0}\frac{f(a+h)-f(a)}{h}$$

则对于某精度条件下，选择适当的$h$值，即可将导数表示为

$$\begin{aligned}
f'(a)\approx&\frac{f(a+h)-f(a)}{h}\\
\approx&\frac{f(a)-f(a-h)}{h}\\
\approx&\frac{f(a+h)-f(a-h)}{2h}\\
\end{aligned}
$$

如果我们得到的是一组一一对应的$x,y$值，那么可以对这组数据进行多项式插值，然后再求解多项式的导数。Lagrange插值函数的表达式为

$$
L_n(x)=\displaystyle\sum_{k=0}^ny_k\frac{\omega_{n+1}(x)}{(x-x_k)\omega'_{n+1}(x_k)}
$$

其中，

$$
\omega_{n+1}(x)=(x-x_0)(x-x_1)\ldots(x-x_n)$$

其两点形式为

$$
L_2(x)=y_1\frac{x-x_2}{x_1-x_2}+y_2\frac{x-x_1}{x_2-x_1}
$$

由此可得两点插值的导数为

$$
L_2'(x)=\frac{y_1}{x_1-x_2}+\frac{y_2}{x_2-x_1}=\frac{y_2-y_1}{h}
$$

显然这是个常量，相当于根本没插值，直接用的前向或者后向差分。

Lagrange插值的三点表达式为

$$
L_3(x)=\frac{(x-x_1)(x-x_2)}{(x_0-x_1)(x_0-x_2)}f(x_0)+\frac{(x-x_0)(x-x_2)}{(x_1-x_0)(x_1-x_2)}f(x_1)+\frac{(x-x_0)(x-x_1)}{(x_2-x_1)(x_2-x_0)}f(x_2)
$$

其导数为

$$
L_3'(x)=\frac{2x-x_1-x_2}{(x_0-x_1)(x_0-x_2)}f(x_0)+\frac{2x-x_0-x_2}{(x_1-x_0)(x_1-x_2)}f(x_1)+\frac{2x-x_0-x_1}{(x_2-x_1)(x_2-x_0)}f(x_2)
$$

由此可得$x_0,x_1,x_2$点的导数分别为

$$\left\{\begin{aligned}
L_3'(x_0)=&\frac{1}{2h}[-3f(x_0)+4f(x_1)-f(x_2)]\\
L_3'(x_1)=&\frac{1}{2h}[-f(x_0)+f(x_2)]\\
L_3'(x_2)=&\frac{1}{2h}[-f(x_0)+4f(x_1)+3f(x_2)]\\
\end{aligned}\right.
$$

此即三点插值下的数值求导公式，通过同样的构造方式可以获取更多点数插值下的求导公式，在此便不再详述了。

## Newton-Cotes公式

定积分的定义式具有十分鲜明的数值特征

$$
\int_{a}^{b}f(x)\text{d}x=\lim_{n\to\infty}\displaystyle\sum_{k=1}^{n}f(a+\frac{k}{n}(b-a))\frac{b-a}{n}
$$

当$n\not\to\infty$时，令$h=\frac{b-a}{n}$，则上式变为

$$
\int_{a}^{b}f(x)\text{d}x\approx\displaystyle\sum_{k=1}^{n}f(a+kh)h
$$

可以十分方便地写成

```julia
# 无脑的数值积分算法
function nativeInt(f,a,b,n)
    h = (b-a)/n
    integ = 0
    for k = 1:n
        integ += f(a+k*h)*h
    end
    return integ
end
```

验证

```julia
julia> include("integration.jl");
julia> f(x) = 2*x^2 + 1     #定义被积函数
julia> nativeInt(f,0,1,20)  #积分区间为[0,1]，积分值为5/3
1.7175000000000002          #当步长为20的时候，误差很大

julia> nativeInt(f,0,1,100) #步长为100时，误差降到了1%
1.6766999999999999
```

可见其精度不怎么样，让牛顿那些人手算100次也的确有些强人所难了，所以牛顿他们就想办法在提高精度的前提下降低运算量。

考虑到定积分定义式的特点，令$x_k=a+k\frac{(b-a)}{n}$，为函数上的某些节点，则积分值可以表示为积分节点函数值$f(x_k)$的加权和，其权系数取决于当前点的斜率，定义为求积系数$A_k$。

$$
\int_{a}^{b}f(x)\text{d}x\approx\displaystyle\sum_{k=1}^{n}A_kf(x_k)
$$

如果$f(x_k)$为已经确定形式的多项式，由于多项式积分具有精确的表达式，所以可以确定最佳的$A_k$使得计算过程简化至最小计算量并且有最佳精度。于是我们马上想到了插值算法，如果令$L_n(x)$为插值函数，则可构造原积分的近似积分

$$
I_n=\int_a^bL_n(x)\text{d}x
$$

其中，

$$
L_n(x)=\displaystyle\sum_{k=0}^nf(x_k)l_k(x)
$$


通过插值积分来构造的插值型积分公式

$$
I_n=(b-a)\displaystyle\sum_{k=0}^{n}C^{(n)}_kf(x_k)
$$

被称为$\textbf{Newton-Cotes公式}$，其中$C^{(n)}_k$为$\textbf{Cotes系数}$。

将lagrange插值函数代入可得

$$
(b-a)\displaystyle\sum_{k=0}^{n}C^{(n)}_kf(x_k)=\displaystyle\sum_{k=0}^nf(x_k)\int_a^bl_k(x)\text{d}x
$$

即

$$\begin{aligned}
&&(b-a)C^{(n)}_k&=\int_a^bl_k(x)\text{d}x\\
\to&&C^{(n)}_k&=\frac{1}{b-a}\int_a^b\frac{\omega_{n+1}(x)}{(x-x_k)\omega'_{n+1}(x_k)}\text{d}x\\
&&\xrightarrow{x=a+hz}&=\int_0^n\frac{\omega_{n+1}(a+hz)}{(a+hz-x_k)\omega'_{n+1}(x_k)}\text{d}x\\

\end{aligned}
$$

其中$x_k$如前文所定义$x_k=a+k\frac{(b-a)}{n}=a+hk$，

$$\left\{\begin{aligned}
\omega_{n+1}(x)&=(x-x_0)(x-x_1)\ldots(x-x_n)\\
\omega'_{n+1}(x_k)&=(x_k-x_0)\ldots(x_k-x_{k-1})(x_k-x_{k+1})\ldots(x_k-x_n)
\end{aligned}\right.
$$

化简可得

$$
C^{(n)}_k=\frac{(-1)^{n-k}}{nk!(n-k)!}\int_0^n\frac{\Omega_{n+1}(z)}{z-k}\text{d}z
$$

其中

$$
\Omega_{n+1}(z)=(z-0)(z-1)\ldots(z-n)
$$

所以，数值积分被转化成求Cotes系数的问题，若将Cotes系数进行数值求解，那么有三个问题需要解决，一是阶乘，二是$\Omega$函数，三是多项式积分公式。

其中，阶乘几乎是我们在学习编程语言过程中，学习递归的第一个案例，几乎没有人没写过，没有什么太大的花样。

```julia
# 阶乘
function factorial(n)
    return  n<=1 ? 1 : n*factorial(n-1)
end
```

$\Omega$函数在此前的插值算法中曾经写过，不过在Newton-Cotes公式中，其输入为一个自然数而非一组系数，所以需要做一些改进。对于多项式的表示则无需改变，通过一个数组来表示多项式即可。注意多项式除法的输出为两项，分别是值和余数。

```julia
# 多项式乘法，形式为an*x^n，大端小端自定义，输出和输入的格式相同
# 建议a[1]为最低位常数项
function polyMulti(a,b)
    na = length(a); nb = length(b)
    c = zeros(na+nb-1)
    for i = 1:na
        c[i:i+nb-1] += a[i]*b
    end
    return c
end
#将(x-x1)(x-x2)...(x-xn)化成an*x^n形式,A[1]为最低位(常数项)
#输入为[x1,x2,...,xn]输出为[a1,a2,...,an]
function polySimplify(x)
    if length(x) == 1
        return [-x[1],1]
    else
        return polyMulti([-x[1],1],polySimplify(x[2:end]))
    end
end
```

多项式积分亦只需变动一位，即

$$
\int a_kx^k\text{d}x=\frac{a_k}{k+1}x^{k+1}
$$

易写为

```julia
#多项式积分，输入为([a0,a1,...,an],aStart,aEnd)
#输出为[b0,b1,...,bn+1]
function polyInf(a,as,ae)
    #因为积分后位数升1，所以幂数从1开始
    return sum([(ae^i-as^i)*a[i]/i for i in 1:length(a)])    
end
```

至此，我们就能够得到Cotes系数

```julia
# 获取Cotes系数
function getCotes(n,k=[])
    if k == []
        return [getCotes(n,i) for i in 0:n]
    end
    kerr = polySimplify([i for i in 0:n if i != k])
    cotes = (-1)^(n-k)/n/factorial(k)/factorial(n-k)
    return cotes*polyInf(kerr,0,n)
end
```

验证

```julia
julia> for n = 1:5
       print(getCotes(n))
       print('\n')
       end
[0.5, 0.5]
[0.16666666666666663, 0.6666666666666667, 0.16666666666666663]
[0.125, 0.375, 0.375, 0.125]
[0.0777777777777775, 0.3555555555555566, 0.13333333333333286, 0.3555555555555543, 0.07777777777777779]
[0.06597222222222247, 0.26041666666666285, 0.17361111111110858, 0.17361111111110858, 0.26041666666666285, 0.06597222222222172]
```

与Cotes系数表比较，可见结果是对的。

|n||
|-|-|-|-
|1|$\frac{1}{2}\quad\frac{1}{2}$
|2|$\frac{1}{6}\quad\frac{4}{6}\quad\frac{1}{6}$
|3|$\frac{1}{8}\quad\frac{3}{8}\quad\frac{3}{8}\quad\frac{1}{8}$
|4|$\frac{7}{90}\quad\frac{16}{45}\quad\frac{2}{15}\quad\frac{16}{45}\quad\frac{7}{90}$
|5|$\frac{19}{288}\quad\frac{25}{96}\quad\frac{25}{144}\quad\frac{25}{144}\quad\frac{25}{96}\quad\frac{19}{288}$

将Cotes系数代入公式

$$
I_n=(b-a)\displaystyle\sum_{k=0}^{n}C^{(n)}_kf(x_k)
$$

则当$n=1$时，可得到梯形公式 

$$I_1=(b-a)\frac{b-a}{2}[f(a)+f(b)]$$

当$n=2$时，得到$\textbf{Simpson公式}$

$$I_2=(b-a)\frac{b-a}{6}[f(a)+4f(\frac{a+b}{2})+f(b)]$$

当$n=4$时，被特称为$\textbf{Cotes公式}$

$$I_4=(b-a)\frac{b-a}{90}
[7f(x_0)+32f(x_1)+12f(x_2)+32f(x_3)+7f(x_4)]$$

则对于任意阶的Newton-Cotes公式，可以写为

```julia
# Newton-Cotes公式
function NewtonCotes(f,a,b,n)
    cotes = getCotes(n)
    return sum([cotes[i]*f(a+(b-a)/n*(i-1)) for i in 1:n+1])
end
```

验证

```julia
julia> include("integration.jl");
julia> f(x) = x^2+2*x+3;
julia> NewtonCotes(f,0,1,3)
4.333333333333333
```

## 复化求积法

Lagrange插值法所存在的问题在于，高阶插值可能会导致局部最优，但其他区域误差很大，基于Lagrange插值的Newton-Cotes公式也有类似的特征，当阶数较大时会导致求解结果不稳定。此外，不同阶数的Newton-Cotes公式针对于不同阶数的多项式拟合精度不同，对于指数函数的无穷阶数，当拟合区间较大时，必然存在精度上的灾难。

```julia
julia> f(x) = x^2+2*x+3;
julia> NewtonCotes(f,0,1,30)
9.752372644513475e25
julia> NewtonCotes(exp,0,10,5)-(exp(10)-1) 
-19714.53098399712
```

解决Lagrange插值中Runge现象的方法是样条插值，其基本思想是对定义域进行分段，在每个小段进行低阶拟合，其总拟合效果要优于对所有区域的高阶拟合。同理，对于Newton-Cotes公式也可以对积分区域进行细分，从而用更低阶的Cotes系数在更狭小的区域中进行求积，这种方法叫做复化求积法。

二阶复化求积法，即复化梯形公式为

$$
T_n=\displaystyle\sum^{n-1}_{k=0}\frac{h}{2}[f(x_k)+f(x_{k+1})]=\frac{h}{2}[f(a)+2\displaystyle\sum^{n-1}_{k=1}f(x_k)+f(b)]$$

记子区间$[x_k,x_{k+1}]$的中点为$x_{k+\frac{1}{2}}$，则复化Simpson公式为

$$\begin{aligned}
S_n&=\displaystyle\sum^{n-1}_{k=0}\frac{h}{6}[f(x_k)+4f(x_{k+\frac{1}{2}})+f(x_{k+1})]\\
&=\frac{h}{6}[f(a)+4\displaystyle\sum^{n-1}_{k=0}f(x_{k+\frac{1}{2}})+2\displaystyle\sum^{n-1}_{k=0}f(x_k)+f(b)]
\end{aligned}
$$

更高阶的Newton-Cotes公式亦然。

复化公式的好处是通过缩减区间长度，可以提高数据精度，因为对于连续函数来说，越短的区间意味着这个区间内的映射关系更趋近于线性。问题在于更细的区段划分意味着更大的运算量，所以实际应用的过程中往往求一步看一步，当不满足精度要求的时候尽量缩短步长，在满足精度之后则进行输出。

对于复化梯形公式来说，对于任意一段积分区间$[x_k,x_{k+1}]$，当我们对其进行二分时，其实只是增加了一个中点，则这个子区间上的积分值为

$$
\frac{h}{4}[f(x_k)+2f(x_{k+\frac{1}{2}})+f(x_{k+1})]
$$

则整个区间的求和公式变为

$$\begin{aligned}
T_{2n}&=\frac{h}{4}\displaystyle\sum^{n-1}_{k=0}[f(x_k)+f(x_{k+1})]+\frac{h}{2}\displaystyle\sum^{n-1}_{k=0}f(x_{k+\frac{1}{2}})\\
&=\frac{1}{2}Tn+\frac{h}{2}\displaystyle\sum^{n-1}_{k=0}f(x_{k+\frac{1}{2}})
\end{aligned}
$$

其Julia实现为

```julia
# 复化梯形公式,f为函数，a,b为积分区间，err为要求精度
function Trapz(f,a,b,err=0.01,nMax=10)
    n = 2
    nMax = 2^nMax
    h = (b-a)/2
    T = h/2*(f(a)+f(b)+2*f(a+h))
    myErr = err + 1                 #初始误差大于目标误差

    while myErr > err && n < nMax
        newX = [2*i-1 for i in 1:n]         #新的x序列
        n = n*2; h = h/2                    #更新n，h值
        new  = sum(f.(a.+h*newX))*h
        myErr = abs(T/2-new)
        T = T/2 + new           #新的梯形值
    end
    return T,myErr
end
```

测试一下指数函数，结果表明当划分区间为$2^10$时，其拟合精度小于1，优于Newton-Cotes公式。

```julia
julia> include("integration.jl");
julia> Trapz(exp,0,10)
(22025.640837203784, 0.5251238525834196)
julia> exp(10)-1
22025.465794806718
```

## Romberg算法

梯形复化求积虽然缩减了求积区间，从而极大提高了求积精度，但同时牺牲Newton-Cotes公式中的高阶项。最直观的方法当然是对Simpson公式或者Cotes公式进行复化求积，但从算法复用的角度考虑，如果能够基于梯形复化求积法构造出高阶Newton-Cotes公式，那将是即为便利的。

对比Simpson公式和梯形求积公式

$$\begin{aligned}
T=&(b-a)\frac{b-a}{2}[f(a)+f(b)]\\
\to T_2=&(b-a)\frac{b-a}{4}[f(a)+f(b)+2f(\frac{a+b}{2})]\\
S=&(b-a)\frac{b-a}{6}[f(a)+4f(\frac{a+b}{2})+f(b)]\\
\end{aligned}
$$

解得

$$
S_n = \frac{4T_2-T}{3}\to S_n=\frac{4T_{2n}-T_n}{3}
$$

同理可得复化Cotes公式和$\textbf{Romberg公式}$

$$\begin{aligned}
C_n =&\frac{16T_{2n}-T_n}{15}\\
R_n =&\frac{64C_{2n}-1_n}{63}\\
\end{aligned}
$$

所以，我们可以直接对梯形复化公式进行修改

```julia
# 复化梯形公式,f为函数，a,b为积分区间，err为要求精度
# flag为1，2，3，4分别表示梯形、Simpson、Cotes、Romberg算法
function Trapz(f,a,b,flag=1,err=0.01,nMax=10)
    flag = min(flag,nMax)       #
    nMax = 2^nMax

    n = 1;    h = b-a                       #初始化n,h
    T = h/2*(f(a)+f(b))
    Tn = zeros(flag)                        #存储梯形公式的值
    Tn[end] = T
    myErr = err + 1                         #初始误差大于目标误差

    while myErr > err && n < nMax
        newX = [2*i-1 for i in 1:n]         #新的x序列
        n = n*2; h = h/2                    #更新n，h值
        new  = sum(f.(a.+h*newX))*h
        myErr = abs(T/2-new)
        T = T/2 + new                       #新的梯形值
        if flag > 1
            Tn[1:end-1] = Tn[2:end]
        end
        Tn[end] = T                         #更新梯形公式
    end

    if flag > 1
        Sn = (4*Tn[2:end]-Tn[1:end-1])/3
        Tn = [Tn[end],Sn[end]]
    end

    if flag > 2
        Cn = (16*Sn[2:end]-Sn[1:end-1])/15  #Cotes公式
        Tn = vcat(Tn,Cn[end])
    end

    if flag > 3
        Rn = (64*Cn[2:end]-Cn[1:end-1])/63  #Romberg公式
        Tn = vcat(Tn,Rn[end])
    end

    return Tn,myErr
end
```

验证

```julia
julia> include("integration.jl");
julia> Trapz(exp,0,10,4)[1]
4-element Array{Float64,1}:
 22025.640837203784
 22025.46579591959
 22025.465794806754
 22025.46579480671
julia> exp(10)-1
22025.465794806718
```

可见在相同积分步长的前提下，Romberg公式的精度优于Cotes优于Simpson优于梯形公式。


# 方程求根

## 二分法

对于方程$f(x)=0$，如果$f(a)\cdot f(b)<0$，则说明$a,b$之间必有根，如果按照某个步长对其进行搜索，那么最终一定能够不断缩小根的取值范围，不断精确化。

二分法就是基于这种思想的一种算法，其实质是对区间进行快速选取。如果已知$f(a)\cdot f(b)<0$，则快速选取$c=\frac{a+b}{2}$，如果$f(a)\cdot f(c)<0$，则说明根在$(a,c)$之间，否则即说明根在$(a,b)$之间，然后不断第迭代下去。

```julia
# equations.jl
# 二分法求根，f为函数，err为误差
# 要求f(a)*f(b)<0
function Dichotomy(f,a,b,err)
    c = (a+b)/2
    val = f(c)
    #如果f(c)和f(a)异号，则将c赋值给b，否则将c赋值给a
    val*f(a)<0 ? b=c : a=c
    #如果大于误差，则继续二分，否则输出c
    return abs(val)>err ? Dichotomy(f,a,b,err) : c
end
```

验证

```julia
julia> include("equations.jl");
julia> f(x) = x^5+x^4+x^2+1;
julia> x = Dichotomy(f,-2,1,0.01);
julia> f(x)
0.009531426464285175
```

## Newton法

如果$f(x)$=0可以写成$x=\varphi(x)$的形式，则可以将这个方程的根理解为$y=\varphi(x)$与$y=x$的交点。现给定一个初值$x_0$，将其代入$\varphi(x)$中，则$\varphi(x_0)$对应一个新的$x$，记为$x_1$。

如果数列$\{x_n\}$收敛，那么必有$x=\lim_{k\to\infty}x_k$即为方程的根。所以，问题的关键在构造合适的$\varphi(x)$使得数列能够快速收敛。

其最简单的形式莫过于$x_{k+1}=x_k+f(x_k)$，不过这种迭代公式看起来十分危险，如果$f(x_k)是一个大于0的单调递增函数，那么结果就爆炸了。

所以，最好为$f(x_k)$乘以一个系数，当$f(x_k)$增加时使之变小，当其递减时使之变大。所以一个比较好的形式为

$$
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}
$$

此即Newton公式。

除了迭代形式，Newton公式也可以作为二分法的一个升级版本，$x_k-\frac{f(x_k)}{f'(x_k)}$可以表示过$(x_k,f(x_k))$做切线与$x$轴的交点，通过切线选取新的$x$值取代了二分法的盲动性。

其Julia实现为

```julia
# Newton法
# f为函数，x为初始值，err为误差
function Newton(f,x,err)
    fx = f(x)
    df = (fx-f(x-err))/err      #近似斜率
    xNew = x-fx/df

    # 如果满足精度要求则返回，否则继续迭代
    return abs(xNew-x) > err ? Newton(f,xNew,err) : xNew
end
```

验证

```julia
julia> include("equations.jl");
julia> f(x) = x^5+x^4+x^2+1;
julia> x = Newton(f,1,0.01);
julia> f(x)
-0.0028617855019934524
```

# 常微分方程

## Euler方法

一阶常微分方程的一般形式为

$$\left\{\begin{aligned}
\frac{\text{d}y}{\text{d}x}=& f(x,y) \to \text{d}y=f(x,y)\text{d}x\\
y(x_0)=&y_0
\end{aligned}\right.
$$

如果将$x$划分为一系列子区间，其端点序列为$x_0,x_1,\ldots,x_n$，则上式可写为

$$\begin{aligned}
\text{d}y_n=&f(x_n,y_n)\text{d}x_n\\
\to y_{n+1}=&y_n+f(x_n,y_n)\text{d}x_n
\end{aligned}
$$

此即$\textbf{Euler公式}$，记$f_n=f(x_n,y_n),h=dx_n$，则可更加简洁地写为

$$y_{n+1}=y_n+f_nh$$


在已知初值的情况下，可以递推地计算出此后的函数值。

```julia
# Euler法
# f为dy/dx=f(x,y), x为输入区间，n为步长
function Euler(f,x,y0,n)
    h = (x[2]-x[1])/n
    x = [x[1]+i*h for i in 1:n]
    y = zeros(n)
    y[1] = y0
    for i in 2:n
        y[i] = y[i-1]+h*f(x[i],y[i-1])
    end
    return y
end
```

验证

```julia
julia> include("equations.jl")        
julia> f(x,y) = y-2*x/y     #导函数
julia> Y(x) = sqrt(1+2*x)   #原函数
julia> x = [0,1]            #区间
julia> y0 = 1
julia> yEuler = Euler(f,x,y0,100);
julia> X = [i for i in 0.01:0.01:1]
julia> yReal = Y.(X)
julia> using Plots
julia> plot(X,[yEuler,yReal],st=[:line :scatter])
julia> savefig("naiveEuler.png")
```

![欧拉拟合](\img\naiveEuler.png)

如果对于[0,1]区间，我们给的初值为$y(1)=y_1$，那么其求解过程将变成从后向前的运算，公式$y_{n+1}=y_n+f(x_n,y_n)\text{d}x_n$也将从显式变成隐式，因为此时的$y_n$将是一个未知量。

这种隐式的Euler公式即为Euler公式的后退格式，如果想要将其改为从前向后的计算风格，即可写为

$$
y_{n+1}=y_n+f(x_{n+1},y_{n+1})\text{d}x_n
$$

如果同时考虑前向格式和后向格式，那么可以将二者综合为

$$
y_{n+1}=y_n+(f_n+f_{n+1})\frac{h}{2}
$$

此即Euler公式的梯形格式，这仍然是一个隐式，所以需要对其进行迭代求值，在给定初值的情况下，有

$$\left\{\begin{aligned}
y_{n+1}^{(0)}=&y_n+hf(x_n,y_n)\\
y_{n+1}^{(k+1)}=&y_n+\frac{h}{2}[f(x_n,y_n)+f(x_{n+1},y_{n+1}^{(k)})]
\end{aligned}\right.
$$

这个算法在处理每个点时，都需要进行大量的迭代，开销很大，但实现过程并不复杂，只需对原始的Euler公式稍加改进即可

```julia
# Euler法
# f为dy/dx=f(x,y), x为输入区间，n为步长
# nIter 为梯形格式的迭代次数
function Euler(f,x,y0,n,nIter)
    h = (x[2]-x[1])/n
    x = [x[1]+i*h for i in 1:n]
    y = zeros(n)
    y[1] = y0
    for i in 2:n
        y[i] = y[i-1]+h*f(x[i],y[i-1])
        for j in 1:nIter
            y[i] = y[i-1]+h*f(x[i],y[i])
        end
    end
    return y
end
```

当迭代次数为1时，被特别称为改进的Euler法。

$$\left\{\begin{aligned}
y_{test}=&y_n+hf(x_n,y_n)\\
y_{n+1}=&y_n+\frac{h}{2}[f(x_n,y_n)+f(x_{n+1},y_{test})]
\end{aligned}\right.
$$

## Runge-kutta法

Runge-Kutta法是一种高精度的单步法，其根源是$Tylor$公式：

$$
y(x_{n+1})=y(x_n)+hy'(x_n)+\frac{h^2}{2!}y''(x_n)+\frac{h^3}{3!}y''(x_n)+\ldots
$$

其中导数之间存在递推关系

$$\left\{\begin{aligned}
y'=&f,\\
y''=&\frac{\partial f}{\partial x}+f\frac{\partial f}{\partial y}\\
y'''=&\frac{\partial^2f}{\partial x^2}+2f\frac{\partial^2f}{\partial x\partial y}+f^2\frac{\partial^2f}{\partial y^2}+\frac{\partial f}{\partial y}(\frac{\partial f}{\partial x}+\frac{\partial f}{\partial y})\\
\vdots
\end{aligned}\right.
$$

Euler法截取了一阶Taylor公式作为拟合方程，如果提高Taylor格式的阶数，将有助于计算结果精度的提高。

现假设有一点$p\in(x_n,x_{n+1})$，且$x_p=x_n+ph,\quad p\leqslant1,h=x_{n+1}-x_n$。设$x_n,x_{n+1}$点的斜率分别为$K_1,K_2$，则$\frac{y_{n+1}-y_n}{h}$可以表示为$K_1,K_2$的线性组合，即

$$
y_{n+1}=y_n+h(\lambda_1K_1+\lambda_2K_2)
$$

由于$x_n$点是暴露给我们的，我们可以设$K_1=f(x_n,y_n)$，并且通过$K_1$来估测$x_p$处的值，即$x_p$处的函数值为$F(x_p)=y_n+K_1ph$，从而可以得到$K_2=f(x_p,y_n+K_1ph)$。

对$K_2$进行展开可以得到

$$\begin{aligned}
K_2 =& f(x_n+ph,y_n+phK_1)=f(x_n,y_n)+\frac{\partial f}{\partial x}\text{d}x+\frac{\partial f}{\partial y}\text{d}y\\
=&f(x_n,y_n)+f_x(x_n,y_n)ph+f_y(x_n,y_n)phK_1\\
\xRightarrow{K_1 = f(x_n,y_n)}&f(x_n,y_n)+ph(f_x(x_n,y_n)+f_y(x_n,y_n)f(x_n,y_n)
\end{aligned}
$$

这正与Taylor展开式有相同的形式，将$K_1,K_2$代入$y_{n+1}=y_n+h(\lambda_1K_1+\lambda_2K_2)$，则可得到

$$\left\{\begin{aligned}
\lambda_1+\lambda_2=1\\
\lambda_2p=\frac{1}{2}    
\end{aligned}\right.
$$

在此条件下，可整理得出二阶Runge-Kutta的表达形式

$$\left\{\begin{aligned}
y_{n+1}=&y_n+h(\lambda_1K_1+\lambda_2K_2)\\
K_1=&f(x_n,y_n)\\
K_2=&f(x_p,y_n+phK_1)\\
\end{aligned}\right.
$$

通过Julia写为

```julia
# Runge-Kutta法，p取1/2
# f为dy/dx=f(x,y), x为输入区间[x1,x2]，n为步长
function RungeKutta(f,x,y0,n)
    h=(x[2]-x[1])/n
    x = [x[1]+i*h for i in 1:n]
    y = zeros(n)
    y[1] = y0
    for i in 2:n
        K1 = f(x[i-1],y[i-1])
        K2 = f(x[i-1]+h/2,y[i-1]+h*K1/2)
        y[i] = y[i-1]+h*K2
    end
    return y
end
```

验证其验证过程与Euler法完全相同，甚至连保存的图片都几乎一致，只需把`Euler`函数换为`RungeKutta`即可。

结合Taylor公式可知，二阶Runge-Kutta法绝非优化的极限，我们可以通过进一步增加参数，来拟合Taylor公式中的高阶项，于是有了更高阶的Runge-Kutta法。

由

$$
y'''=\frac{\partial^2f}{\partial x^2}+2f\frac{\partial^2f}{\partial x\partial y}+f^2\frac{\partial^2f}{\partial y^2}+\frac{\partial f}{\partial y}(\frac{\partial f}{\partial x}+\frac{\partial f}{\partial y})
$$

若三阶Runge-Kutta项的形式为

$$\begin{aligned}
y_{n+1}=&y_n+h(\lambda_1K_1+\lambda_2K_2+\lambda_3K_3)\\
K_1=&f(x_n,y_n)\\
K_2=&f(x_p,y_n+phK_1)\\
K_3=&f(x_n+qh,y_n+qh(rK_1+sK_2))
\end{aligned}
$$

则可推导出各参数所需要满足的条件

$$\left\{\begin{aligned}
\lambda_2p+\lambda_3p=\frac{1}{2}\\
\lambda_2p^2+\lambda_3q^2=\frac{1}{3}\\
\lambda_3pqs=\frac{1}{6}
\end{aligned}\right.
$$

故可选取
$$\begin{aligned}
y_{n+1}=&y_n+\frac{h}{6}(K_1+4K_2+K_3)\\
K_1=&f(x_n,y_n)\\
K_2=&f(x_n+\frac{h}{2},y_n+\frac{h}{2}K_1)\\
K_3=&f(x_n+h,y_n-hK_1+2hK_2)
\end{aligned}
$$

四阶Runge-Kutta亦然，其比较常用的形式为

$$\begin{aligned}
y_{n+1}=&y_n+\frac{h}{6}(K_1+2K_2+2K_3+K_4)\\
K_1=&f(x_n,y_n)\\
K_2=&f(x_n+\frac{h}{2},y_n+\frac{h}{2}K_1)\\
K_3=&f(x_n+\frac{h}{2},y_n+\frac{h}{2}K_2)\\
K_3=&f(x_n+h,y_n+hK_2)
\end{aligned}
$$

```julia
# Runge-Kutta法
# f为dy/dx=f(x,y), x为输入区间[x1,x2]，n为步长
# deg为阶数,支持2，3，4阶
# 二阶需要输入参数p,lmd1,lmd2;三阶需要额外输入q,r,s,lmd3
# 四阶需要另输入a,b,c,d,lmd4
function RungeKutta(f,x,y0,n;deg=2;kwargs...)
    h=(x[2]-x[1])/n
    x = [x[1]+i*h for i in 1:n]
    y = zeros(n)
    y[1] = y0

    
    # 初始化参数
    if deg == 2
        p,lmd1,lmd2 = isempty(kwargs) ? [1/2,1/2,1/2] : [
            kwargs[:p],kwargs[:lmd1],kwargs[:lmd2]]
    elseif deg == 3
        p,q,r,s,lmd1,lmd2,lmd3 = isempty(kwargs) ? [
            1/2,1,-1,2,1/6,2/3,1/6] : [kwargs[:p],
            kwargs[:q],kwargs[:r],kwargs[:s],
            kwargs[:lmd1],kwargs[:lmd2],kwargs[:lmd3]]
    elseif deg == 4
        p,q,r,s,a,b,c,d,lmd1,lmd2,lmd3 = isempty(kwargs) ? [
            1/2,1/2,0,0,1,0,1,0,1/6,1/3,1/3,1/6] : [
            kwargs[:p],kwargs[:q],kwargs[:r],kwargs[:s],
            kwargs[:a],kwargs[:b],kwargs[:c],kwargs[:d],
            kwargs[:lmd1],kwargs[:lmd2],kwargs[:lmd3],kwargs[:lmd4]]
    end

    for i in 2:n
        K1 = f(x[i-1],y[i-1])
        K = K1
        if deg>1
            K2 = f(x[i-1]+p*h,y[i-1]+p*h*K1)        #参数p
            K = lmd1*K1+lmd2*K2
        end
        if  deg>2
            K3 = f(x[i-1]+q*h,y[i-1]+q*h*(r*K1+s*K2))#参数q,r,s
            K += lmd3                   #参数lmd3
        end
        if deg>3
            K4 = f(x[i-1]+a*h,y[i-1]+a*h*(b*K1+c*K2+d*K3))
            K += lmd4
        end
        y[i] = y[i-1]+h*K
    end
    return y
end
```
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

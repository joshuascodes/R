Build a function to perform KFold CV
================
Joshua Freimark
01/2019

  - R version 4.0.0 (2020-04-24)

  - Build a function that performs K-fold cross validation.

Simulate data

``` r
n=200;p=500
x=matrix(rnorm(n*p),n,p)
b=c(rep(1,5),rep(0,p-5))

y=1+x%*%b+rnorm(n)
length(y); dim(x)   
```

    ## [1] 200

    ## [1] 200 500

Implementing LASSO

``` r
library(glmnet)
model.lasso=glmnet(x,y)      #Set alpha=1  for LASSO. By default the function glmnet will fit a lasso estimate
hb=coef(model.lasso,s=0.2)   #s = Lambda value. We will try 0.2 to start out
#hb
```

Implementing ridge

``` r
model.ridge=glmnet(x,y,alpha=0)         #alpha=0 indicates the ridge penalty
hb=coef(model.ridge,s=0.2)              #The tuning parameter (lambda) is represented by `s` in this function
#hb
```

Choosing lambda via cross validation. Begin with lambda=0.01

``` r
dat=data.frame(x=x,y=y)
la.grid=seq(0.01,1,length.out=100)
la.grid
```

    ##   [1] 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15
    ##  [16] 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30
    ##  [31] 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.40 0.41 0.42 0.43 0.44 0.45
    ##  [46] 0.46 0.47 0.48 0.49 0.50 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.60
    ##  [61] 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75
    ##  [76] 0.76 0.77 0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90
    ##  [91] 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.00

Compute the corresponding cross validation error (cv.err) for the
associated Lambda value

``` r
k=5     #no.of fols
n=length(dat$y)
ncv=ceiling(n/k)
cv.ind.f=rep(seq(1:k),ncv)
cv.ind=cv.ind.f[1:n]
cv.ind.random=sample(cv.ind,n,replace=F)

cv.err=c()
```

The cross validation function

``` r
for(l in 1:length(la.grid)){
  MSE=c()
  for(j in 1:k){
    train=dat[cv.ind.random!=j,]
    response=train$y
    design=as.matrix(train[,names(dat)!="y"])
    mod=glmnet(design,response,lambda=la.grid)
    hb=coef(mod,s=la.grid[l])
    test=dat[cv.ind.random==j,]
    resp.test=test$y
    fitted.values=cbind(1,as.matrix(test[,names(dat)!="y"]))%*%hb
    MSE[j]=mean((resp.test-fitted.values)^2)
  }
  cv.err[l]=mean(MSE)
}
```

Graph displaying the CV error associated with each lambda value

``` r
plot(la.grid, cv.err,type="l")   
```

![](Build-a-function-to-perform-KFold-CV_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
best.la=la.grid[which.min(cv.err)]
best.la        #Returns the optimal value for lambda
```

    ## [1] 0.05

Using the optimal value of lambda selected from the CV function to
estimate LASSO

``` r
model=glmnet(x,y)
final.hb=coef(model,s=best.la)
final.hb
```

    ## 501 x 1 sparse Matrix of class "dgCMatrix"
    ##                         1
    ## (Intercept)  0.9660371656
    ## V1           0.8313345619
    ## V2           0.9528468441
    ## V3           0.9041186290
    ## V4           0.9329635211
    ## V5           0.9646886988
    ## V6           .           
    ## V7           .           
    ## V8           .           
    ## V9           0.0245024662
    ## V10          .           
    ## V11          0.0603410966
    ## V12          .           
    ## V13          .           
    ## V14          .           
    ## V15          .           
    ## V16          .           
    ## V17          0.0511793838
    ## V18          .           
    ## V19          .           
    ## V20          .           
    ## V21          .           
    ## V22         -0.0860500737
    ## V23          .           
    ## V24          .           
    ## V25          .           
    ## V26          0.0831602314
    ## V27          .           
    ## V28          .           
    ## V29         -0.0402183564
    ## V30          .           
    ## V31          .           
    ## V32          .           
    ## V33          .           
    ## V34          .           
    ## V35          .           
    ## V36          .           
    ## V37          .           
    ## V38          .           
    ## V39          .           
    ## V40          .           
    ## V41         -0.0133523934
    ## V42          .           
    ## V43          .           
    ## V44          .           
    ## V45          .           
    ## V46          .           
    ## V47         -0.1098616172
    ## V48          .           
    ## V49          .           
    ## V50          .           
    ## V51          .           
    ## V52          .           
    ## V53          .           
    ## V54          .           
    ## V55          .           
    ## V56          .           
    ## V57          .           
    ## V58          .           
    ## V59          .           
    ## V60          .           
    ## V61          .           
    ## V62         -0.1591749148
    ## V63          .           
    ## V64          .           
    ## V65          .           
    ## V66          0.0012868254
    ## V67          .           
    ## V68          .           
    ## V69          0.0199941324
    ## V70         -0.0304669776
    ## V71          .           
    ## V72         -0.0135283213
    ## V73          .           
    ## V74          0.0072736746
    ## V75          .           
    ## V76          .           
    ## V77          .           
    ## V78          .           
    ## V79          .           
    ## V80         -0.0170567844
    ## V81          .           
    ## V82          .           
    ## V83         -0.0357700347
    ## V84          .           
    ## V85          0.0205280109
    ## V86          .           
    ## V87          .           
    ## V88          .           
    ## V89         -0.0430827204
    ## V90          .           
    ## V91         -0.0451523171
    ## V92          .           
    ## V93          .           
    ## V94          .           
    ## V95          .           
    ## V96          .           
    ## V97         -0.0481711900
    ## V98          .           
    ## V99          .           
    ## V100         .           
    ## V101         .           
    ## V102         .           
    ## V103        -0.0020004783
    ## V104         .           
    ## V105        -0.0698868646
    ## V106         .           
    ## V107         .           
    ## V108         .           
    ## V109         0.0306218924
    ## V110         .           
    ## V111         .           
    ## V112         .           
    ## V113        -0.0494677724
    ## V114         .           
    ## V115         .           
    ## V116         .           
    ## V117         .           
    ## V118         0.0217484202
    ## V119        -0.1220380485
    ## V120         .           
    ## V121         .           
    ## V122         .           
    ## V123         .           
    ## V124         .           
    ## V125         .           
    ## V126         .           
    ## V127         .           
    ## V128         .           
    ## V129         .           
    ## V130        -0.0665311449
    ## V131         .           
    ## V132         .           
    ## V133         .           
    ## V134         .           
    ## V135         .           
    ## V136         .           
    ## V137         .           
    ## V138         .           
    ## V139         0.0430596476
    ## V140         .           
    ## V141         .           
    ## V142         .           
    ## V143         .           
    ## V144        -0.1782097112
    ## V145         0.0170431272
    ## V146         .           
    ## V147         .           
    ## V148         .           
    ## V149         .           
    ## V150         .           
    ## V151         .           
    ## V152         .           
    ## V153         .           
    ## V154         .           
    ## V155        -0.0388467037
    ## V156         .           
    ## V157         .           
    ## V158         .           
    ## V159         .           
    ## V160         .           
    ## V161         .           
    ## V162         .           
    ## V163         .           
    ## V164         0.1017466122
    ## V165         .           
    ## V166         0.0025627122
    ## V167         .           
    ## V168         .           
    ## V169         0.0723864345
    ## V170         .           
    ## V171         .           
    ## V172         .           
    ## V173         .           
    ## V174         .           
    ## V175         .           
    ## V176         .           
    ## V177         .           
    ## V178         .           
    ## V179         .           
    ## V180         .           
    ## V181         0.0005446348
    ## V182         .           
    ## V183        -0.0137751902
    ## V184         .           
    ## V185         .           
    ## V186         .           
    ## V187        -0.0521764650
    ## V188         .           
    ## V189         .           
    ## V190         0.0418367624
    ## V191         .           
    ## V192         .           
    ## V193         .           
    ## V194         .           
    ## V195         .           
    ## V196         .           
    ## V197         .           
    ## V198         .           
    ## V199         .           
    ## V200         .           
    ## V201         .           
    ## V202         .           
    ## V203         0.0168913100
    ## V204        -0.0493319435
    ## V205         0.0100949776
    ## V206         .           
    ## V207         0.0536347792
    ## V208         .           
    ## V209         .           
    ## V210         .           
    ## V211         .           
    ## V212         .           
    ## V213         .           
    ## V214         .           
    ## V215         .           
    ## V216         .           
    ## V217         .           
    ## V218        -0.0219470390
    ## V219         .           
    ## V220         .           
    ## V221         .           
    ## V222         .           
    ## V223         .           
    ## V224         .           
    ## V225         .           
    ## V226         .           
    ## V227         0.0097987382
    ## V228         .           
    ## V229        -0.0195592103
    ## V230         .           
    ## V231         .           
    ## V232         .           
    ## V233         .           
    ## V234         .           
    ## V235         .           
    ## V236         .           
    ## V237         .           
    ## V238         .           
    ## V239         .           
    ## V240         .           
    ## V241         .           
    ## V242         .           
    ## V243         0.0161416262
    ## V244         .           
    ## V245         .           
    ## V246         .           
    ## V247         .           
    ## V248         .           
    ## V249         .           
    ## V250         .           
    ## V251         .           
    ## V252         .           
    ## V253         .           
    ## V254         .           
    ## V255         .           
    ## V256         .           
    ## V257         .           
    ## V258        -0.0043602381
    ## V259         .           
    ## V260         .           
    ## V261         .           
    ## V262         .           
    ## V263         .           
    ## V264         .           
    ## V265         .           
    ## V266         .           
    ## V267         .           
    ## V268        -0.0839735672
    ## V269         .           
    ## V270         .           
    ## V271         .           
    ## V272         .           
    ## V273         .           
    ## V274         0.0170665045
    ## V275         .           
    ## V276         .           
    ## V277         0.0158882187
    ## V278         .           
    ## V279         .           
    ## V280         .           
    ## V281         .           
    ## V282         .           
    ## V283         .           
    ## V284         0.0537683011
    ## V285        -0.0428414275
    ## V286         0.0613374415
    ## V287         .           
    ## V288        -0.0358055596
    ## V289         .           
    ## V290         .           
    ## V291         .           
    ## V292         .           
    ## V293         .           
    ## V294         .           
    ## V295         .           
    ## V296         .           
    ## V297         .           
    ## V298         .           
    ## V299         .           
    ## V300         .           
    ## V301         .           
    ## V302        -0.0038606053
    ## V303         .           
    ## V304         .           
    ## V305         0.0114589377
    ## V306         0.0370384671
    ## V307         .           
    ## V308         0.0551744803
    ## V309         .           
    ## V310         .           
    ## V311         .           
    ## V312         .           
    ## V313         .           
    ## V314         0.0678213537
    ## V315         0.0230837824
    ## V316         .           
    ## V317         .           
    ## V318         .           
    ## V319         .           
    ## V320         .           
    ## V321         .           
    ## V322         .           
    ## V323         0.0404642635
    ## V324         .           
    ## V325         .           
    ## V326         .           
    ## V327         .           
    ## V328         0.0599756165
    ## V329         .           
    ## V330         .           
    ## V331         .           
    ## V332        -0.0230601861
    ## V333         .           
    ## V334         .           
    ## V335         .           
    ## V336         .           
    ## V337         0.0779138257
    ## V338         .           
    ## V339         .           
    ## V340         .           
    ## V341         .           
    ## V342         .           
    ## V343         .           
    ## V344         .           
    ## V345         .           
    ## V346         .           
    ## V347        -0.0149667869
    ## V348         .           
    ## V349         .           
    ## V350         .           
    ## V351         .           
    ## V352        -0.0383431297
    ## V353         .           
    ## V354        -0.0885994512
    ## V355         0.0007902425
    ## V356         .           
    ## V357         0.0788515962
    ## V358         .           
    ## V359         .           
    ## V360         .           
    ## V361         .           
    ## V362         .           
    ## V363         .           
    ## V364         .           
    ## V365         .           
    ## V366        -0.0320831488
    ## V367         .           
    ## V368         0.0110913385
    ## V369         .           
    ## V370         .           
    ## V371         .           
    ## V372         .           
    ## V373         .           
    ## V374         0.0871401037
    ## V375         .           
    ## V376        -0.0414097962
    ## V377         .           
    ## V378         .           
    ## V379         .           
    ## V380         .           
    ## V381         .           
    ## V382         .           
    ## V383         .           
    ## V384         0.1702686026
    ## V385         .           
    ## V386         0.0502864522
    ## V387         .           
    ## V388         .           
    ## V389         0.0296218580
    ## V390         .           
    ## V391        -0.0738089527
    ## V392         0.0948451527
    ## V393         .           
    ## V394        -0.1492232399
    ## V395         .           
    ## V396         .           
    ## V397         .           
    ## V398         .           
    ## V399         .           
    ## V400         .           
    ## V401         0.1238261915
    ## V402         .           
    ## V403         .           
    ## V404         .           
    ## V405         .           
    ## V406         .           
    ## V407         .           
    ## V408         .           
    ## V409         .           
    ## V410        -0.0124773294
    ## V411         .           
    ## V412         .           
    ## V413         .           
    ## V414         .           
    ## V415         0.0225272100
    ## V416         .           
    ## V417         .           
    ## V418         .           
    ## V419         .           
    ## V420         .           
    ## V421         .           
    ## V422         0.0175588019
    ## V423         .           
    ## V424         0.0256406096
    ## V425         .           
    ## V426         .           
    ## V427         .           
    ## V428         .           
    ## V429         .           
    ## V430         .           
    ## V431         .           
    ## V432         .           
    ## V433         .           
    ## V434        -0.0206413934
    ## V435         .           
    ## V436         .           
    ## V437         .           
    ## V438        -0.0649328721
    ## V439         .           
    ## V440         0.0588666430
    ## V441         .           
    ## V442         .           
    ## V443         .           
    ## V444         .           
    ## V445         .           
    ## V446         .           
    ## V447         .           
    ## V448         .           
    ## V449         .           
    ## V450         .           
    ## V451         .           
    ## V452         .           
    ## V453         0.0060284326
    ## V454         .           
    ## V455        -0.0116423048
    ## V456         .           
    ## V457        -0.0106476631
    ## V458         0.1010509128
    ## V459         .           
    ## V460         .           
    ## V461         .           
    ## V462         .           
    ## V463         0.0714673395
    ## V464         .           
    ## V465         .           
    ## V466        -0.0166958871
    ## V467         .           
    ## V468         .           
    ## V469        -0.0394819985
    ## V470         .           
    ## V471         .           
    ## V472         .           
    ## V473         .           
    ## V474         .           
    ## V475         .           
    ## V476         .           
    ## V477         .           
    ## V478         0.0856588509
    ## V479         .           
    ## V480         .           
    ## V481         .           
    ## V482         .           
    ## V483         .           
    ## V484         .           
    ## V485         .           
    ## V486         .           
    ## V487         .           
    ## V488         .           
    ## V489         .           
    ## V490         .           
    ## V491         .           
    ## V492         .           
    ## V493        -0.0156931433
    ## V494         .           
    ## V495         .           
    ## V496         .           
    ## V497         .           
    ## V498         .           
    ## V499         .           
    ## V500         .

Using the built in function glmnet to compare if the function we created
returns similar results as glmnet

``` r
cv.mod=cv.glmnet(x,y,nfolds = 5,lambda = seq(0.01,1,length.out = 100))
cv.mod$lambda.min  #Should return a value close to the value of best.la
```

    ## [1] 0.07

-----

Logistic regression to estimate if the S\&P500 will move up or down.

``` r
library(ISLR)
data(Smarket)

logit.model=glm(Direction~Lag1+Lag2+Lag3+Lag4+Volume,
                data=Smarket,family=binomial)

summary(logit.model)
```

    ## 
    ## Call:
    ## glm(formula = Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Volume, 
    ##     family = binomial, data = Smarket)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -1.448  -1.202   1.068   1.145   1.317  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)
    ## (Intercept) -0.124785   0.240659  -0.519    0.604
    ## Lag1        -0.073126   0.050164  -1.458    0.145
    ## Lag2        -0.042360   0.050084  -0.846    0.398
    ## Lag3         0.010869   0.049929   0.218    0.828
    ## Lag4         0.009058   0.049947   0.181    0.856
    ## Volume       0.134660   0.158312   0.851    0.395
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1731.2  on 1249  degrees of freedom
    ## Residual deviance: 1727.6  on 1244  degrees of freedom
    ## AIC: 1739.6
    ## 
    ## Number of Fisher Scoring iterations: 3

``` r
glm.probs=predict(logit.model,type="response")
glm.probs[1:10]
```

    ##         1         2         3         4         5         6         7         8 
    ## 0.4944585 0.4844253 0.4878925 0.5156843 0.5097410 0.5044691 0.4900052 0.5107911 
    ##         9        10 
    ## 0.5160232 0.4882519

``` r
glm.pred=rep("Down",1250)
glm.pred[glm.probs>0.5]="Up"
glm.pred
```

    ##    [1] "Down" "Down" "Down" "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Down"
    ##   [11] "Down" "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Down"
    ##   [21] "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Down" "Up"   "Up"   "Up"  
    ##   [31] "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Down" "Down" "Down"
    ##   [41] "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Up"   "Up"  
    ##   [51] "Up"   "Down" "Down" "Down" "Up"   "Up"   "Down" "Up"   "Up"   "Up"  
    ##   [61] "Down" "Down" "Up"   "Down" "Down" "Down" "Down" "Down" "Down" "Down"
    ##   [71] "Up"   "Up"   "Up"   "Down" "Down" "Down" "Up"   "Down" "Up"   "Up"  
    ##   [81] "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Down" "Down"
    ##   [91] "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down"
    ##  [101] "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [111] "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Down" "Up"  
    ##  [121] "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"  
    ##  [131] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Down" "Up"  
    ##  [141] "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down"
    ##  [151] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Down"
    ##  [161] "Up"   "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [171] "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Down" "Down"
    ##  [181] "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"  
    ##  [191] "Up"   "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Down" "Down" "Up"  
    ##  [201] "Up"   "Up"   "Down" "Down" "Down" "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [211] "Down" "Down" "Up"   "Up"   "Down" "Up"   "Up"   "Down" "Down" "Up"  
    ##  [221] "Up"   "Up"   "Down" "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [231] "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Down" "Down"
    ##  [241] "Down" "Down" "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [251] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [261] "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Down"
    ##  [271] "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down"
    ##  [281] "Down" "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Down"
    ##  [291] "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [301] "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [311] "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [321] "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [331] "Down" "Down" "Up"   "Down" "Down" "Up"   "Up"   "Down" "Up"   "Up"  
    ##  [341] "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down"
    ##  [351] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"  
    ##  [361] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [371] "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [381] "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Down" "Down" "Up"   "Up"  
    ##  [391] "Up"   "Up"   "Down" "Down" "Down" "Down" "Up"   "Up"   "Down" "Down"
    ##  [401] "Up"   "Down" "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [411] "Down" "Up"   "Up"   "Up"   "Down" "Down" "Down" "Down" "Up"   "Up"  
    ##  [421] "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"  
    ##  [431] "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down"
    ##  [441] "Down" "Down" "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Down"
    ##  [451] "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Down" "Up"   "Up"   "Up"  
    ##  [461] "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Down" "Down" "Up"  
    ##  [471] "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [481] "Up"   "Down" "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Down"
    ##  [491] "Down" "Up"   "Up"   "Up"   "Down" "Down" "Down" "Down" "Up"   "Up"  
    ##  [501] "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [511] "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Down" "Up"   "Up"   "Up"  
    ##  [521] "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"  
    ##  [531] "Up"   "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [541] "Up"   "Up"   "Up"   "Down" "Down" "Down" "Down" "Up"   "Up"   "Down"
    ##  [551] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"  
    ##  [561] "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Down"
    ##  [571] "Down" "Down" "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Down" "Up"  
    ##  [581] "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [591] "Up"   "Up"   "Down" "Down" "Down" "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [601] "Down" "Up"   "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Down" "Down"
    ##  [611] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [621] "Down" "Down" "Down" "Down" "Up"   "Up"   "Up"   "Down" "Up"   "Up"  
    ##  [631] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"  
    ##  [641] "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Down" "Down" "Up"  
    ##  [651] "Up"   "Down" "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Down"
    ##  [661] "Down" "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [671] "Down" "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [681] "Up"   "Down" "Up"   "Down" "Down" "Up"   "Down" "Up"   "Up"   "Up"  
    ##  [691] "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [701] "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Up"  
    ##  [711] "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [721] "Up"   "Down" "Down" "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [731] "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Down" "Up"  
    ##  [741] "Up"   "Up"   "Down" "Down" "Down" "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [751] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [761] "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [771] "Up"   "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [781] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [791] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Up"  
    ##  [801] "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Down" "Down" "Up"   "Up"  
    ##  [811] "Up"   "Down" "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [821] "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [831] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [841] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"  
    ##  [851] "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [861] "Down" "Up"   "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [871] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Up"   "Up"  
    ##  [881] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [891] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Up"  
    ##  [901] "Up"   "Up"   "Down" "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Down"
    ##  [911] "Down" "Down" "Up"   "Up"   "Down" "Down" "Down" "Up"   "Up"   "Up"  
    ##  [921] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [931] "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"  
    ##  [941] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Up"  
    ##  [951] "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"   "Down"
    ##  [961] "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"  
    ##  [971] "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [981] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ##  [991] "Up"   "Down" "Down" "Down" "Up"   "Down" "Down" "Down" "Down" "Up"  
    ## [1001] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down"
    ## [1011] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1021] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1031] "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Down" "Up"   "Up"   "Up"  
    ## [1041] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1051] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1061] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1071] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1081] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1091] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1101] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1111] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1121] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1131] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1141] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1151] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1161] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1171] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1181] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1191] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1201] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1211] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1221] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Down" "Up"   "Up"   "Up"  
    ## [1231] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"  
    ## [1241] "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"   "Up"

Confusion matrix

``` r
table(glm.pred,Smarket$Direction)
```

    ##         
    ## glm.pred Down  Up
    ##     Down  146 136
    ##     Up    456 512

Another example building a cross validation function

``` r
set.seed(1234)
n=202
x=rnorm(n)
y=-1-0.5*x+1*x^2+rnorm(n)
```

``` r
dat=data.frame(y=y,x=x)

k=5   #no. of folds
n=length(dat$y)
ncv=ceiling(n/k); ncv
```

    ## [1] 41

``` r
cv.ind.f=rep(seq(1:k),ncv)
#cv.ind.f

cv.ind=cv.ind.f[1:n]
length(cv.ind)
```

    ## [1] 202

``` r
cv.ind.random=sample(cv.ind,n,replace=F)
#cv.ind.random
```

Build the cross validation function

``` r
cv.function=function(dat,k){
  n=length(dat$y)
  if(k>n){stop("check no. of folds")}
  ncv=ceiling(n/k); 
  ##ceiling function is the smallest integer>x
  ##Example: 4.2-->5, 4.7-->5
  cv.ind.f=rep(seq(1:k),ncv)
  #cv.ind.f
  cv.ind=cv.ind.f[1:n]
  #length(cv.ind)
  cv.ind.random=sample(cv.ind,n,replace=F)
  #cv.ind.random
  MSE=c()
  for(j in 1:k){
    train=dat[cv.ind.random!=j,]
    response=train$y
    design=as.matrix(train[,names(dat)!="y"])
    mod=lm(response~design)
    hb=coef(mod)
    test=dat[cv.ind.random==j,]
    resp.test=test$y
    fitted.values=cbind(1,as.matrix(test[,names(dat)!="y"]))%*%hb
    MSE[j]=mean((resp.test-fitted.values)^2)}
  cv.err=mean(MSE)
  #cv.err
  return(cv.err)}
```

Models to choose from. Select the model with the lowest cvm (which is
the MSE)

``` r
cvm=c()
dat1=data.frame(y=y,x1=x)
cvm[1]=cv.function(dat1,k=5)

dat2=data.frame(y=y,x1=x,x2=x^2)
cvm[2]=cv.function(dat2,k=5)

dat3=data.frame(y=y,x1=x,x2=x^2,x3=x^3)
cvm[3]=cv.function(dat3,k=5)

dat4=data.frame(y=y,x1=x,x2=x^2,x3=x^3,x4=x^4)
cvm[4]=cv.function(dat4,k=5)

dat5=data.frame(y=y,x1=x,x2=x^2,x3=x^3,x4=x^4,x5=x^5)
cvm[5]=cv.function(dat5,k=5)

dat6=data.frame(y=y,x1=x,x2=x^2,x3=x^3,x4=x^4,x5=x^5,x6=x^6)
cvm[6]=cv.function(dat6,k=5)

cvm
```

    ## [1] 3.338297 1.036925 1.043529 1.077325 1.167861 1.412818

Forward selection

``` r
library(leaps)
regfit=regsubsets(y~., data=dat6, method="forward",nvmax=10)
s=summary(regfit)
s$adjr2
```

    ## [1] 0.5995153 0.6890295 0.6883059 0.6873254 0.6888908 0.6873049

-----

## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax
for authoring HTML, PDF, and MS Word documents. For more details on
using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that
includes both content as well as the output of any embedded R code
chunks within the document. You can embed an R code chunk like this:

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

## Including Plots

You can also embed plots, for example:

![](Build-a-function-to-perform-KFold-CV_files/figure-gfm/pressure-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.

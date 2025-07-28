# 원스토어 인앱 결제 연동 가이드

## 원스토어 In-App SDK
원스토어 인앱결제는 다양한 결제수단과 강력한 보안 그리고 편리하고 안전한 결제 서비스를 제공합니다.

- 최신버전 인앱결제 가이드
[원스토어 인앱결제 API V7 (SDK V21) 안내 및 다운로드](#원스토어-인앱결제-api-v7sdk-v21-연동-안내-및-다운로드)

## 원스토어 인앱결제 API V7(SDK V21) 연동 안내 및 다운로드
원스토어의 최신 인앱결제 API V7(SDK V21)이 출시되었습니다.
보다 강력하고 다양한 기능을 지원하는 최신 버전을 적용해보세요.

- API V4(SDK V16) 이하 버전과는 호환되지 않습니다. 인앱결제 API V4(SDK V16)에 대한 안내 및 다운로드는 여기를 클릭해주세요.
- 현재 판매중인 앱을 대한민국 외 국가/지역으로 배포하기 위해서는 아래 가이드를 참고해주세요
  - [대한민국 외 국가 및 지역 배포를 위한 가이드](https://onestore-dev.gitbook.io/dev/tools/glb)

If you are comfortable with English, please change the language to English from the upper left side in this page.

### 01. 원스토어 인앱결제 개요

#### 원스토어 인앱결제란?
원스토어 인앱결제(In-App Purchase, IAP)는 안드로이드 앱 내에 구현된 상품을 원스토어의 인증 및 결제 시스템을 이용하여 사용자에게 판매, 청구하여 개발자에게 정산하는 서비스입니다.

인앱 상품 결제를 위해 원스토어 서비스(ONE store service, OSS) 앱과 연동이 필요하며, 원스토어 서비스 앱은 원스토어 결제 서버와 연동하여 인앱상품의 결제를 수행합니다.

원스토어 인앱결제를 적용하기 위해서는 '원스토어 IAP SDK(In-App Purchase Software Development Kit)'를 적용하면 됩니다.

#### 인앱 상품의 유형
원스토어 인앱결제 API V7(SDK V21)는 관리형 상품과 구독형 상품. 두 가지 유형의 인앱 상품을 제공 합니다.
우선, 각 타입이 어떤 특성을 가지고 있는지 확인하신 후 본인이 제공하는 상품에 맞게 인앱 상품 목록을 구성해 보시기 바랍니다.

- 관리형 상품
  - 구매 후 소비(consume)하기 전까지 재구매가 되지 않는 상품입니다. 해당 상품을 다시 구매하기 위해서는 소비(consume) 처리를 해야 합니다.
  - 해당 특성을 이용하여 소비성/영구성/기간제 형태의 인앱 상품을 제공 할 수 있습니다.

- 월정액 상품 (Deprecated)
  - 한 번 가입하면 지정된 날짜에 매월 일정 금액이 자동결제 되는 상품으로, 자동결제 갱신(재결제)은 원스토어에서 처리합니다.
  - SDK V21 (API V7) 적용 이후 신규 월정액 상품은 만들 수 없습니다. (기존에 만들어진 월정액 상품은 계속 이용 하실 수 있습니다.) 
  - 구독형 상품을 이용하여 정기 결제 상품을 제공할 수 있습니다,  

- 구독형 상품
  - SDK V21 (API V7) 부터 제공되는 정기 결제 상품 타입입니다.
  - 정해진 주기에 따라 원스토어에서 정기 결제를 처리하며, 신규 고객을 유치하기 위한 프로모션 기능 및 결제 수단 변경 등의 사용자 편의 기능을 제공합니다. 
  - 구독형 상품에 대한 자세한 내용은 정기결제 적용하기 페이지를 참고하시기 바랍니다.

#### 결제 프로세스
원스토어 인앱결제는 크게 다음의 네 가지 프로세스로 구성되어 있습니다. 

##### 원스토어 로그인 하기
###### GaaSignInClient 초기화
GaaSignInClient는 원스토어 로그인을 하기 위한 라이브러리입니다.

getClient() 통해 인스턴스를 생성합니다.

```java
  val signInClient = GaaSignInClient.getClient(activity)
```

###### 포그라운드 로그인하기
slientSignIn()과 달리 해당 함수는 UiThread 에서만 호출해야 합니다.

기본적으로 먼저 백그라운드 로그인을 시도하지만 실패에 대한 처리를 SDK에서 전담하여 처리합니다.
이 후, 로그인 화면을 띄워 사용자에게 로그인을 유도합니다.

'''java
  signInClient.launchSignInFlow(activity) { signInResult ->
  
  }
'''
 
###### 백그라운드 로그인하기
slientSignIn() 통해 백그라운도 로그인을 호출합니다.

사용자가 이미 원스토어 계정에 로그인 되어있을 경우 이후 부터는 백그라운드에서 토큰로그인을 시도합니다. 성공 또는 실패에 대한 결과값으로 SignInResult 객체로 응답을 받습니다.

'''java

  signInClient.silentSignIn { signInResult ->
  
  }
'''

##### 인앱 상품 구매 요청하기 
상품을 구매하려면 먼저 상품 상세정보(queryProductDetailAsync) API 호출하여 상품의 정보을 가지고 launchPurchaseFlow API를 통해 구매 화면을 호출합니다.
결제가 완료되면 PurchaseClient 객체를 초기화할 때 입력한 PurchasesUpdatedListener를 통해 구매정보를 전달받을 수 있습니다.
결과 코드값을 검증한 후에 성공으로 확인되면 인앱 상품에 해당하는 아이템을 지급하는 프로세스를 진행합니다.
아래 그림은 개발사 앱과 원스토어 사이의 인앱 상품을 구매하는 프로세스를 나타낸 것입니다. 

[그림] * 해당 그림은 Sequence Diagram을 Text로 전환하였습니다.

launchPurchaseFlow()
---
1. App -> SDK : startConnection()
2. SDK -> ONE store : bindService()
3. ONE store -> SDK : onServiceConnected()
4. SDK -> ONE store : isBillingSupportedExtParams()
5. ONE store -> SDK : return iapResult
6. SDK -> APP : onSetupFinished()
7. APP -> ONE store : queryProductDetailAsync()
8. ONE store -> App : return List<ProductDetail> (ProductDetailsLister receive data)
9. App -> SDK : launchPurchaseFlow()
10. SDK -> ONE store : getPurchaseInternalExtraParams()
11. ONE store -> SDK : return InApp Intent
12. SDK -> ONE store : start Payment activity
13. ONE store -> SDK : return Purchase data
14. SDK -> App : onPurchaseUpdate()

##### 구매 확인하기 
원스토어 IAP 라이브러리 V6 이상을 사용하는 경우 3일 이내에 구매 확인을 진행해야 합니다. 구매 확인이 되지 않으면, 고객에게 구매 금액이 환불됩니다. 

다음 두 가지 방법을 사용하면 구매를 확인할 수 있습니다. 

###### 소비하기 (consume) 

관리형 상품의 특징은 구매 후 소비(consume) 하기 전까지 재구매가 되지 않는 것입니다. 따라서, 관리형 상품 중 반복 구매가 가능한 소비성 상품을 제공할 경우에는 반드시 PurchaseClient.consumeAsync()를 사용하여 소비 처리를 해야 고객이 해당 상품을 다시 구매할 수 있습니다.  

소비성 형태의 상품을 구현할 때는 구매 프로세스 완료 후 바로 구매한 상품을 소비시키고, 소비가 완료된 후에 아이템을 지급합니다.  

사용자가 앱(게임)을 실행하는 시점 또는 앱(게임)에 로그인하는 시점에 위의 API들을 호출하여, 미지급된 상품이 있는지 사전에 체크하고 상품을 지급하는 프로세스를 추가하는 것을 권장합니다.  

[그림] * 해당 그림은 Sequence Diagram을 Text로 전환하였습니다.

consumeAsync()
---
1. App -> ONE store : queryPurchase()
2. ONE store --> App : return Purchase data
3. App -->> ONE store : consumeAsync()
4. ONE store -->> App : return success / fail
5. App ---> ONE store


###### 구매 확인하기 (acknowledge)

소비성 상품이 아닌 경우, PurchaseClient.acknowledgeAsync() 를 사용합니다. 소비 처리를 하지않고, 구매 인증만 하는 방법으로 영구성 상품을 구현할 수 있습니다. 

영구성 형태의 상품을 구현할 때는, 소비가 아닌 구매 확인 (acknowledge)을 해야 합니다. 

구독 상품의 경우에도 구매 확인이 필요하며, 최초 결제에 대해서만 구매 확인을 진행하면 됩니다.  


네트워크 단절이나 앱의 갑작스러운 비정상 종료 등으로 인해, 고객은 결제를 완료하였으나 개발사에 결제정보 전달이 누락되어 인앱 상품이 지급되지 않는 경우가 발생할 수 있습니다.
구매정보 가져오기(queryPurchases) 및 소비(consume) 기능을 활용하여 구매가 완료 되었으나 미지급된 내역이 있는지 확인 후 미지급된 상품이 지급 처리될 수 있도록 프로세스를 구현하는 것을 권장합니다. 

##### 구독형 상품 관리
구독형 상품(subscription)은 첫 구매 이후 상품 결제 주기에 따라 정기적으로 결제를 갱신합니다.

원스토어는 사용자가 정기 결제 상품을 관리 할 수 있도록 일시 중지, 구독 해지, 결제 수단 변경 등 다양한 기능이 있는 정기 결제 관리 화면을 제공합니다.   

launchManageSubscription API를 호출하거나 딥링크를 사용하여 정기 결제 관리 메뉴를 제공 할 수 있습니다. 

#### 권장 개발 환경
원스토어 IAP SDK를 적용하기 위해 다음과 같은 개발 환경이 필요합니다.

- Android 6.0 이상 버전(API 버전 23 이상)
- Java SDK 11 버전
- Android studio 2.0 이상 버전

### 02. 사전준비

#### 상품 등록하기 
- 상품 등록과 관련된 내용은 아래 가이드를 참고 부탁 드립니다. 
  - [상품 등록과 관리](https://onestore-dev.gitbook.io/dev/docs/apps)

#### 라이선스 키(Public Key) 및 OAuth 인증 정보 확인하기 
공통정보 >  라이선스 관리 메뉴에서 라이선스 키와 서버 API를 위한 OAuth 인증 정보를 확인할 수 있습니다. 

- 라이선스 키 : 원스토어가 전달한 인앱결제 내역의 위변조 여부를 확인하는 용도로 사용합니다.
- OAuth 인증 정보 : 원스토어 서버 API를 사용하기 위한 인증 용도로 사용합니다.

#### 원스토어 인앱결제 Sample App 다운로드
원스토어 인앱결제를 사용하기 위해 필요한 샘플 앱을 깃허브(github)에서 다운로드 받을 수 있습니다. 

#### 인앱결제 라이브러리 추가하기 
프로젝트 최상위 build.gradle 파일에 원스토어 maven 주소를 등록합니다.

<pre><code>
allprojects {
    repositories {
        ...
        maven { url 'https://repo.onestore.net/repository/onestore-sdk-public' }
    }
}
</code></pre>

다음은 앱의 build.gradle 파일에 원스토어 결제 라이브러리 종속 항목을 추가합니다.

<pre><code>
dependencies {
    implementation "com.onestorecorp.sdk:sdk-iap:21.02.01"
    implementation "com.onestorecorp.sdk:sdk-configuration-kr:1.0.0"
}
</code></pre>

#### 원스토어 앱 설치하기 
개발자나 사용자가 원스토어 인앱결제를 이용하기 위해서는 원스토어 앱이 필요합니다.

- [원스토어 앱 다운로드 안내](https://onestore-dev.gitbook.io/dev/policy/download)

### 03. 결제 테스트 및 보안
#### 개요
원스토어는 개발환경(이하 Sandbox) 또는 상용 결제환경(이하 상용테스트)에서 인앱상품의 결제 테스트를 지원합니다.

Sandbox는 상용 결제환경이 아닌 가상의 결제환경으로, 결제화면에서 결제실패 또는 성공 중 원하는 응답을 선택하면 해당 응답결과를 전송합니다.

상용테스트는 상용 결제환경에서 해당 인앱상품의 결제를 진행하여 결제 결과를 전송합니다.

상용테스트 환경에서 결제를 하실 경우 취소하지 않으면 과금이 발생할 수 있으니, 결제 테스트 후 반드시 결제취소를 진행해주셔야 합니다.

- 주의사항
  - 상품을 배포하고자 하는 OS에서 반드시 한 번은 샌드박스 테스트를 진행해야 합니다.
  - OGL AAB/APK의 경우 안드로이드 환경에서 샌드박스 테스트를 진행하면 됩니다. 
  - Sandbox 및 상용환경에서 테스트하기 위해서는 사전에 원스토어 ID를 테스트 ID로 등록해야 합니다. 등록하지 않은 원스토어 ID로 테스트할 경우 Sandbox에서 테스트가 불가능하고, 상용테스트 환경에서는 테스트 상용결제가 아닌 실제 상용결제가 진행되어 과금이 되므로 주의해주시기 바랍니다.
  - 원스토어는 테스트 ID를 통해 결제한 내역에 대해 책임지지 않습니다. 테스트 ID의 관리 및 결제 테스트는 개발사 내부 책임자의 관리감독 하에 진행하시기 바랍니다.

#### 인앱결제 테스트 화면
In-App정보 화면에서 '결제 테스트' 버튼을 누르면 인앱결제 테스트 화면이 제공됩니다.

인앱결제 테스트 화면에서 테스트 ID를 관리하거나, 결제 테스트 내역을 조회할 수 있습니다.

#### 테스트 ID 등록/관리
개발 및 상용환경에서 인앱결제를 테스트하기 위해서는 사전에 테스트 ID를 등록해야 합니다.

인앱결제 테스트 화면에서 '테스트 ID 등록/관리' 탭을 선택하시면, 테스트 ID를 관리하는 화면으로 이동합니다.

테스트 ID는 원스토어 사용자 ID로, 원스토어에 회원으로 등록되어 있는 ID를 조회하여 등록할 수 있습니다. 

만약, 원스토어 회원으로 등록되어 있지 않다면 원스토어 앱을 실행하여 테스트용 원스토어 ID를 만드신 후 해당 ID를 테스트 ID로 등록하면 됩니다.  

##### 테스트 ID 등록하기  
- 테스트 ID로 사용 할 원스토어 회원 ID를 조회합니다.

- 조회한 원스토어 ID 확인 후, 해당 테스트 ID로 어느 결제환경(Sandbox, 상용테스트)에서 결제 테스트를 진행할지 선택합니다.

- 필요한 경우 참고 사항을 입력하고 등록 버튼을 눌러 테스트 ID를 등록 합니다.

- 하나의 테스트 ID는 하나의 결제환경만 선택가능하며, 결제환경 설정은 언제든 변경할 수 있습니다.

- 등록되어 있는 테스트 ID로 어느 환경에서 결제 테스트 할지 설정할 수 있습니다. 

- 테스트 ID 결제환경을 변경할 경우 반드시 '저장' 버튼을 눌러 새로운 설정을 저장해야 합니다.

##### 테스트 ID 삭제하기 (개별 삭제, 일괄 삭제)
- 테스트 ID가 더 이상 유효하지 않을 경우 테스트 ID를 삭제할 수 있습니다. 

- '삭제'하기 버튼을 눌러 개별 테스트 ID를 삭제하거나, 여러 테스트 ID를 체크한 후 '체크한 ID 삭제' 버튼을 눌러 일괄 삭제할 수 있습니다.

##### 테스트 ID 설정 복사
- 동일한 테스트 ID를 여러 앱에 공통으로 사용할 경우, 테스트 ID 설정을 다른 앱으로 복사할 수 있습니다.

- 복사가 가능한 앱만 체크 박스가 활성화 되며, 테스트 ID 설정을 복사할 대상 앱을 선택한 수 '복사하기'를 누르면 됩니다.

- 대상 앱에 바이너리(apk)가 등록되어 있지 않을 경우 Sandbox로 설정된 테스트 ID만 복사됩니다.

#### Sandbox 환경에서의 결제 테스트 (필수) 
##### Sandbox 결제 테스트 및 검증 
Sandbox는 상용 결제환경이 아닌 가상의 결제환경으로, 결제화면에서 결제실패 또는 성공 중 원하는 응답을 선택하면 해당 응답결과를 전송합니다.

Sandbox에서 결제한 내역은 인앱결제 테스트 화면의 'Sandbox' 탭에서 조회할 수 있으며, 결제취소도 가능합니다.

OS 별로 Sandbox에서 테스트를 1회 이상 진행해야 하며, 테스트를 수행하지 않은 경우에는 검증을 요청 할 수 없습니다. 

> 국가/지역 별 테스트는 필요하지 않습니다. 테스트하기 편한 국가/지역에서 테스트하시면 됩니다. 

원게임루프를 통해  AAB/APK를 배포하는 경우에는 안드로이드 OS 테스트 결과를 참조하므로, 안드로이드 환경에서 테스트를 진행하시면 됩니다. 

테스트 환경 (상용 및 Sandbox) 정보는 원스토어 서비스와 연결된 동안 메모리에 유지됩니다. 만약 개발자 센터를 통해 테스트 ID의 테스트 환경을 변경했다면, 반드시 앱을 종료하고 다시 시작해야 합니다.

>참고사항
>- 테스트 ID 설정이 Sandbox로 되어 있어야 Sandbox 결제 테스트가 가능합니다.

##### 구독형 상품의 Sandbox 테스트 
구독형 상품을 개발할 때 실제 시간과 동일하게 결제 주기가 설정된다면, 테스트에 너무 많은 시간이 필요합니다. 

이에 Sandbox 환경에서는 구독 주기별 소요 시간을 조정하여 테스트를 원활하게 할 수 있도록 지원하고 있습니다. 

각 결제 주기 및 기능 별로 Sandbox 내 시간 흐름은 다음과 같습니다. 

[
  {
    "결제 주기": "1주",
    "Sandbox 내 시간": "5분"
  },
  {
    "결제 주기": "5분",
    "Sandbox 내 시간": "5분"
  },
  {
    "결제 주기": "3개월",
    "Sandbox 내 시간": "10분"
  },
  {
    "결제 주기": "6개월",
    "Sandbox 내 시간": "15분"
  },
  {
    "결제 주기": "1년",
    "Sandbox 내 시간": "30분"
  }
]

[
  {
    "기능": "무료 구독",
    "Sandbox 내 시간": "3분"
  },
  {
    "결제 주기": "유예 기간",
    "Sandbox 내 시간": "5분"
  },
  {
    "결제 주기": "계정 보류",
    "Sandbox 내 시간": "10분"
  },
  {
    "결제 주기": "가격 변경 반영 (7일)",
    "Sandbox 내 시간": "5분"
  },
  {
    "결제 주기": "가격 변경 동의 기간 (30일)",
    "Sandbox 내 시간": "5분"
  }
]

##### 상용테스트 환경에서의 결제 테스트 (선택) 
상용테스트는 상용 결제환경에서 해당 인앱상품의 결제를 진행하여 결제 결과를 전송합니다.

상용테스트로 결제한 내역은 인앱결제 테스트 화면의 '상용테스트' 탭에서 조회할 수 있으며, 결제취소도 가능합니다.

상용테스트 환경에서 결제를 하실 경우 취소하지 않으면 과금이 발생할 수 있으니, 결제 테스트 후 반드시 결제취소를 진행해주셔야 합니다.

> 참고사항
> - 테스트 ID 설정이 상용테스트로 되어 있어야 상용테스트 환경에서 결제 테스트가 가능합니다.
> - 상용테스트로 설정하지 않은 테스트 ID 또는 테스트 ID로 등록되어있지 않은 원스토어 ID로 결제할 경우 테스트 상용결제가 아닌 실제 상용결제가 진행되어 과금되니 주의해주시기 바랍니다. 
> - 실제 상용결제가 진행된 경우, 즉시 원스토어 측으로 결제정보를 전달하여 결제취소를 요청하시기 바랍니다.

##### 인앱결제 테스트 결과 확인 및 결제취소
인앱결제 테스트 내역의 결제상태 확인 및 결제취소를 할 수 있습니다. 

Sandbox에서 결제한 내역은 인앱결제 테스트 화면의 'Sandbox' 탭에서, 상용테스트로 결제한 내역은 인앱결제 테스트 화면의 '상용테스트' 탭에서 결제내역을 조회할 수 있습니다.

- Sandbox 환경
Sandbox 환경에서 결제 테스트한 내역이 제공됩니다. 

결제한 상품정보 및 결제상태를 조회할 수 있으며, 결제취소도 가능합니다.

- 상용테스트 환경
상용테스트 환경에서 결제 테스트한 내역이 제공됩니다.

결제한 상품정보 및 결제상태를 조회할 수 있으며, 결제취소도 가능합니다.

##### 보안 및 인증
###### 인앱 결제의 보안
- 원스토어 인앱 결제는 데이터의 위변조 여부를 확인하는 방법으로 서명(Signature)의 유효성 체크 방식을 사용합니다.
모바일 앱은 많은 공격의 위협에 노출되어 있으므로 이러한 위험을 최소화하기 위해 개발사 앱이나 서버에서 서명 인증을 진행하는 것을 권장합니다. 
다음은 좀 더 안전한 인앱결제를 위해 개발사에서 추가로 준비해야 할 사항입니다.

- 개발사 서버 사용
  - APK Reverse Engineering 등을 이용한 공격을 어렵게 하기 위해서 구현 코드와 공개 키(Public key) 등은 서버에서 보관하고 인증을 진행할 수 있도록 합니다.

  - 구매 정보를 저장할 때도 단말기의 저장소보다 개발사 서버의 저장소를 사용하여 아이템의 사용 허가를 확인하도록 하는 것이 좋습니다.

- 변경된 코드 사용

  - 원스토어에서 제공하는 샘플 코드는 많은 사람들에게 오픈 되어 있는 코드로, 원본을 그대로 사용하는 것보다는 수정해서 사용하는 것을 권장합니다. 동일한 코드를 사용하면 그만큼 공격의 위험에 노출되기 쉽습니다.

  - Proguard 같은 코드 난독화 툴을 이용하여 결제 관련 코드를 보호하는 것이 좋습니다.

  - 공개 키라 할지라도 앱 코드 안에 일반 문자열로 넣는 것은 안전한 방법이 아닙니다. 다른 문자열과 XOR하여 쉽게 노출되지 않도록 하는 등 공격자가 쉽게 접근 할 수 없도록 하는 것이 안전합니다.

- 결제 요청 시 ‘developerPayload’ 필드를 활용

  - 개발사는 결제 요청 시 ‘developerPayload’ 필드에 임의의 정보를 넣고 결제 완료 시 다시 전달 받을 수 있습니다. 
해당 필드에 timestamp 등을 조합한 추가적인 보안 검증용 데이터를 넣어 결제 결과에 대한 검증을 좀 더 안전하게 할 수 있습니다.

###### 원스토어 인앱결제 인증 방법
- 인증 준비

  - 인증을 위한 키 알고리즘은 RSA 방식을 사용하며, 서명을 위한 알고리즘으로 ‘SHA512withRSA’를 사용합니다.
서명 검증키는 사전준비 페이지 내 '라이선스 키 및 OAuth 인증 정보 확인' 부분을 참고하시기 바랍니다.
서명 검증키의 값은 단말기나 서버 등 인증을 하는 위치에 따라 적절하게 저장하여 사용합니다.

- 서명 인증 샘플 코드

  - SDK에서는 'AppSecurity' 유틸리티 클래스를 제공하며 'verifyPurchase' 메서드를 이용해 서명 인증을 할 수 있습니다. 
SDK를 이용하지 않고 개발사가 직접 서명 인증을 하려면 'AppSecurity'와 동일한 기능을 하는 코드를 직접 구현해야 합니다. 
자세한 구현 내용은 SDK 라이브러리와 함께 배포하는 샘플을 참고합니다.

### 04. 원스토어 인앱결제 SDK를 사용해 구현하기
#### 개요
원스토어 결제 라이브러리는 Java 환경에서 최신 기능을 제공합니다.
이 가이드에서는 원스토어 결제 라이브러리 기능 구현하는 방법을 설명합니다.

#### 프로젝트 설정
##### 레파지토리 및 종속 항목 추가
프로젝트 최상위 gradle 파일에 원스토어 maven 주소를 등록합니다.

Android Studio (version: bumblebee) 이상에서는 settings.gradle파일에서 추가합니다.

<pre><code>
repositories {
    ... 
    maven { url 'https://repo.onestore.net/repository/onestore-sdk-public' }
}
</code></pre>
앱의 build.gradle 파일에 원스토어 결제 라이브러리 종속 항목을 추가합니다.

<pre><code>
dependencies {
    ...
    def iap_lastest_version = "21.xx.xx"
    implementation "com.onestorecorp.sdk:sdk-iap:$iap_lastest_version"
}
</code></pre>

##### \<queries\> 설정
AndroidManifest.xml 파일에 <queries>를 설정 해야합니다.
자세한 내용은 공지사항을 참조하세요.

> \<queries\> 태그를 설정하지 않으면 SDK에서 원스토어 서비스를 찾을 수 없습니다.

```xml
<manifest>
    <!-- 
        if your binary use ONE store's In-app SDK,
        Please make sure to declare the following query on Androidmanifest.xml. 
        Refer to the notice for more information.
        https://dev.onestore.net/devpoc/support/news/noticeView.omp?noticeId=32968
     -->
    <queries>
        <intent>
            <action android:name="com.onestore.ipc.iap.IapService.ACTION" />
        </intent>
        <intent>
            <action android:name="android.intent.action.VIEW" />
            <data android:scheme="onestore" />
        </intent>
    </queries>
    ...
    <application>
        ...
    </application>
</manifest>
```
##### 스토어 지정을 위한 개발자 옵션 설정
v21.02.00 업데이트 – 글로벌 스토어 선택 기능 추가
 IAP SDK 21.02.00 부터 아래와 같이 onestore:dev_option의 android:value 값을 설정하면, SDK와 연동되는 스토어 앱을 지정 할 수 있습니다.

```xml
<manifest>
    <application>
        <meta-data android:name="onestore:dev_option" android:value="onestore_01" />
    </application>
</manifest>
```

> 값 (android:value)에 대응하는 적용 대상 국가
> - onestore_00 = 대한민국 (South Korea) (기본값)
> - onestore_01 = 싱가포르, 타이완 (Singapore, Taiwan)
> - onestore_02 = 미국 – Digital Turbine (United States)

> 21.01.00 버전에서는 android:value 값이 global만 설정 가능하며, 싱가포르/타이완 스토어 앱만 지정이 가능했습니다. 

> 주의: 배포 버전의 바이너리에서는 이 옵션을 반드시 제거해주세요. 

#### 인앱 라이브러리 적용하기
##### 로그레벨 설정
개발 단계에서 로그 레벨을 설정하여 SDK의 데이터의 흐름을 좀 더 자세히 노출할 수 있습니다.
android.util.Log에 정의된 값을 기반으로 동작합니다.

```android
/**
 * Set the log level.<br/>
 * {@link Log#VERBOSE}, {@link Log#DEBUG}, {@link Log#INFO},
 * {@link Log#WARN}, {@link Log#ERROR}
 * @param level int
 */
com.gaa.sdk.base.Logger.setLogLevel(2)
```

상수에 대응하는 값
- VERBOSE = 2
- DEBUG = 3
- INFO (default) = 4
- WARN = 5
- ERROR = 6

> 배포 빌드 버전에서는 보안에 취약할 수 있으니 이 옵션을 삭제해야 합니다.

##### 오류 처리
원스토어 결제 라이브러리는 IapResult 형식으로 오류를 반환합니다. IapResult에는 앱에서 발생할 수 있는 결제 관련 오류를 분류하는 ResponseCode가 포함되어 있습니다. 예를 들어 RESULT_NEED_LOGIN 이나 RESULT_NEED_UPDATE 같은 오류 코드가 수신되면 앱에서 그에 맞는 처리를 해야 합니다.

##### 원스토어 로그인 하기
- GaaSignInClient 초기화

GaaSignInClient는 원스토어 로그인을 하기 위한 라이브러리입니다.
getClient() 통해 인스턴스를 생성합니다.

```Kotlin
val signInClient = GaaSignInClient.getClient(activity)
```
```Java
GaaSignInClient signInClient = GaaSignInClient.getClient(activity);
```

- 백그라운드 로그인하기

slientSignIn() 통해 백그라운도 로그인을 호출합니다.

사용자가 이미 원스토어 계정에 로그인 되어있을 경우 이후 부터는 백그라운드에서 토큰로그인을 시도합니다. 성공 또는 실패에 대한 결과값으로 SignInResult 객체로 응답을 받습니다.

```Kotlin
signInClient.silentSignIn { signInResult ->
  
}
```
```Java
signInClient.silentSignIn(new OnAuthListener() {
    @Override
    public void onResponse(@NonNull SignInResult signInResult) {
        
    }
});
```

- 포그라운드로 로그인하기

slientSignIn()과 달리 해당 함수는 UiThread 에서만 호출해야 합니다.

기본적으로 먼저 백그라운드 로그인을 시도하지만 실패에 대한 처리를 SDK에서 전담하여 처리합니다.
이 후, 로그인 화면을 띄워 사용자에게 로그인을 유도합니다.

```Kotlin
signInClient.launchSignInFlow(activity) { signInResult ->
  
}
```
```Java
signInClient.launchSignInFlow(activity, new OnAuthListener() {
    @Override
    public void onResponse(@NonNull SignInResult signInResult) {
        
    }
});
```
##### PurchaseClient 초기화
PurchaseClient는 원스토어 결제 라이브러리와 앱 간의 통신을 위한 기본 인터페이스입니다.

단일 이벤트에 관한 여러 개의 PurchasesUpdatedListener 콜백이 발생하는 상황을 피할 수 있도록 PurchaseClient 연결을 열어 두는 것을 권장합니다.

PurchaseClient를 생성하려면 newBuilder()를 사용합니다. 구매 관련 업데이트를 수신하려면 setListener()를 호출하여 PurchasesUpdatedListener에 대한 참조를 전달해야 합니다. 이 리스너는 앱의 모든 구매 관련 업데이트를 수신합니다.
setBase64PublicKey()는 SDK에서 구매 데이터 위변조에 대한 서명 확인 작업을 합니다. 옵션 값이지만 사용하는 것을 권장합니다.

라이선스 키는 '라이선스 관리' 메뉴에서 확인 가능합니다.

> 라이선스 키의 경우, 앱 내 코드로 저장하기 보다는 보안이 될 수 있도록 서버 등을 이용하여 전달받아 사용하는 것을 권장합니다.

```Kotlin
private val listener = PurchasesUpdatedListener { iapResult, purchases ->
       // To be implemented in a later section.
}

private var purchaseClient = PurchaseClient.newBuilder(activity)
   .setListener(listener)
   .setBase64PublicKey(/*your public key*/) // optional
   .build()
```
```Java
private PurchasesUpdatedListener listener = new PurchasesUpdatedListener {
     @Override
     public void onPurchasesUpdated(IapResult iapResult, List<PurchaseData>) {
         // To be implemented in a later section.
     }
};

private PurchaseClient purchaseClient = PurchaseClient.newBuilder(activity)
   .setListener(listener)
   .setBase64PublicKey(/*your public key*/) // optional
   .build();
```

##### 원스토어 서비스 연결 설정
PurchaseClient를 생성 후 원스토어 서비스에 연결해야 합니다.

연결하려면 startConnection()을 호출합니다. 연결 프로세스는 비동기이며 클라이언트 연결이 완료되면 PurchaseClientStateListener를 통해 콜백이 수신됩니다.

다음 예는 연결을 시작하고 사용 준비가 되었는지 테스트하는 방법입니다.

```Kotlin
purchaseClient.startConnection(object : PurchaseClientStateListener {
    override fun onSetupFinished(iapResult: IapResult) {
        if (iapResult.isSuccess) {
            // The PurchaseClient is ready. You can query purchases here.
        }
    }

    override fun onServiceDisconnected() {
        // Try to restart the connection on the next request to
        // PurchaseClient by calling the startConnection() method.
    }
})
```
```Java
purchaseClient.startConnection(new PurchaseClientStateListener {
    @Override
    public void onSetupFinished(IapResult iapResult) {
        if (iapResult.isSuccess()) {
            // The PurchaseClient is ready. You can query purchases here.
        }
    }

    @Override
    public void onServiceDisconnected() {
        // Try to restart the connection on the next request to
        // PurchaseClient by calling the startConnection() method.
    }
});
```

##### 상품 상세정보 조회하기
상품 상세정보를 조회하려면 queryProductDetailsAsync()를 호출합니다. 사용자에게 제품을 표시하기 전에 중요한 단계입니다.

queryProductDetailsAsync()를 호출할 때 setProductType()과 함께 원스토어 개발자 센터에서 생성된 인앱 상품 ID 문자열 목록을 지정하는 ProductDetailParams의 인스턴스를 전달합니다.

ProductType은 아래와 같습니다.

Product에 대응되는 Enum
- 관리형 상품 = ProductType.INAPP
- 정기 결제 상품 (구독 상품) = ProductType.SUBS
- 월 정액 상품 = ProductType.AUTO (이 상품은 향후 지원되지 않을 예정입니다.) 

위의 모든 유형의 데이터를 한 번에 조회하고 싶을 경우는 ProductType.ALL 설정하면 됩니다.

> ProductType.ALL은 상품 상세정보 조회하기에서만 사용할 수 있으며, 구매 요청하기, 구매 내역 조회하기 에서는 사용할 수 없습니다.

비동기 작업 결과를 처리하려면 ProductDetailsListener 인터페이스를 구현해야 합니다.

```Kotlin
val params = ProductDetailsParams.newBuilder()
        .setProductIdList(productIdList)
        .setProductType(ProductType.INAPP)
        .build()
purchaseClient.queryProductDetailsAsync(params) { iapResult, productDetails -> 
    // Process the result.
}
```
```Java
ProductDetailsParams params = ProductDetailsParams.newBuilder()
        .setProductIdList(productIdList)
        .setProductType(ProductType.INAPP)
        .build();

purchaseClient.queryProductDetailsAsync(params, new ProductDetailsListener() {
    @Override
    public void onProductDetailsResponse(IapResult iapResult, List<ProductDetail>) {
        // Process the result. 
    }
});
```

##### 구매 요청하기
앱에서 구매 요청을 하기 위해서는 기본 스레드에서 launchPurchaseFlow() 함수를 호출합니다.

이 함수는 queryProductDetailsAsync() 함수를 호출해서 얻은 ProductDetail 객체의 값을 토대로 PurchaseFlowParams 객체를 생성합니다.
PurchaseFlowParams 객체를 생성하려면 PurchaseFlowParams.Builder 클래스를 사용합니다.

setDeveloperPayload()는 개발사에서 임의로 입력한 값으로 최대 200byte입니다. 이 값은 결제 후에 데이터의 정합성과 부가 데이터를 확인하기 위해 사용할 수 있습니다.
setProductName()은 상품 이름을 결제 시 변경하여 노출하고 싶을 때 사용됩니다.
setQuantity()는 관리형 인앱 상품에만 적용되며 한 상품을 여러 개 구매할 때 사용됩니다.

> 원스토어는 사용자에게 할인 쿠폰, 캐시백 등의 다양한 혜택 프로모션을 진행하고 있습니다.
개발사는 구매 요청 시에 gameUserId, promotionApplicable 파라미터를 이용하여 앱을 사용하는 유저의 프로모션 참여를 제한하거나 허용할 수 있습니다.
개발사는 앱의 고유한 유저 식별 번호 및 프로모션 참여 여부를 선택하여 전달하고, 원스토어는 이 값을 토대로 사용자의 프로모션 혜택을 적용하게 됩니다.

> gameUserId, promotionApplicable 파라미터는 옵션 값으로 원스토어 사업 부서 담당자와 프로모션에 대해 사전협의가 된 상태에서만 사용하여야 하며, 일반적인 경우에는 값을 보내지 않습니다.
또한, 사전협의가 되어 값을 보낼 경우에도 개인 정보보호를 위해 gameUserId는 hash된 고유한 값으로 전송하여야 합니다.

```Kotlin
val activity: Activity = ...

val purchaseFlowParams = PurchaseFlowParams.newBuilder()
      .setProductId(productId)
      .setProductType(productType)
      .setDeveloperPayload(devPayload)    // optional
      .setQuantity(1)                     // optional
      .setProductName("")                 // optional
      .setGameUserId("")                  // optional
      .setPromotionApplicable(false)      // optional
      .build()

purchaseClient.launchPurchaseFlow(activity, purchaseFlowParams)
```
```Java
Activity activity = ...

PurchaseFlowParams purchaseFlowParams = PurchaseFlowParams.newBuilder()
      .setProductId(productId)
      .setProductType(productType)
      .setDeveloperPayload(devPayload)    // optional
      .setQuantity(1)                     // optional
      .setProductName("")                 // optional
      .setGameUserId("")                  // optional
      .setPromotionApplicable(false)      // optional
      .build();

purchaseClient.launchPurchaseFlow(activity, purchaseFlowParams);
```

launchPurchaseFlow() 호출에 성공하면 아래와 같은 화면을 표시합니다. [그림 1]은 정기 결제 구매 화면을 나타냅니다.

(그림 1) : 호출이 성공하였을때 결제 구매 화면이 실제로 단말에 표시되는 것을 단말에서 캡쳐한 화면

구매에 성공하면 PurchasesUpdatedListener 인터페이스의 onPurchasesUpdated() 함수에 구매 작업 결과가 전송됩니다. 이 리스너는 PurchaseClient 초기화 할 때 setListener() 함수를 사용하여 지정됩니다.

```Kotlin
override fun onPurchasesUpdated(iapResult: IapResult, purchases: List<PurchaseData>?) {
    if (iapResult.isSuccess && purchases != null) {
        for (purchase in purchases) {
            handlePurchase(purchase)
        }
    } else if (iapResult.responseCode == ResponseCode.NEED_UPDATE) {
        // PurchaseClient by calling the launchUpdateOrInstallFlow() method.
    } else if (iapResult.responseCode == ResponseCode.NEED_LOGIN) {
        // PurchaseClient by calling the launchLoginFlow() method.
    } else {
        // Handle any other error codes.
    }
}
```
```Java
@Override
public void onPurchasesUpdated(IapResult iapResult, List<PurchaseData> purchases) {
    if (iapResult.isSuccess() && purchases != null) {
        for (purchase in purchases) {
            handlePurchase(purchase);
        }
    } else if (iapResult.getResponseCode() == ResponseCode.NEED_UPDATE) {
        // PurchaseClient by calling the launchUpdateOrInstallFlow() method.
    } else if (iapResult.getResponseCode() == ResponseCode.NEED_LOGIN) {
        // PurchaseClient by calling the launchLoginFlow() method.
    } else {
        // Handle any other error codes.
    }
}
```

(그림 2) : 결제가 완료되었을때의 화면을 단말에서 캡쳐한 화면

구매가 성공하면 구매 데이터에는 사용자 및 상품 ID를 나타내는 고유 식별자인 구매 토큰도 생성됩니다. 구매 토큰을 앱 내에 저장할 수 있지만 구매를 인증하고 사기로부터 보호할 수 있는 백엔드 서버로 토큰을 전달하는 것이 좋습니다.

관리형 상품과 정기 결제 상품의 구매 토큰은 결제가 일어날 때마다 구매 토큰이 발행됩니다. (월정액 상품의 경우 자동 결제가 갱신되는 동안 구매 토큰은 동일하게 유지됩니다.)

또한 사용자는 영수증 번호가 포함된 거래 영수증을 이메일로 받습니다. 관리형 상품은 구매할 때마다 이메일을 받으며, 월정액 상품과 정기 결제 상품은 처음 구매 시 그리고 이후 갱신될 때마다 이메일을 받습니다.

##### 정기 결제
정기 결제는 취소될 때까지 자동으로 갱신됩니다. 정기 결제는 다음 상태를 가질 수 있습니다.

- 활성: 사용자가 콘텐츠 사용에 문제가 없는 양호한 상태이며 정기 결제에 접근할 수 있습니다.

- 일시 중지 예약: 사용자가 정기 결제를 이용 중 일시 중지를 하고 싶을 때 선택할 수 있습니다.

  - 주간 정기 결제: 1~3주 단위로 일시 중지할 수 있습니다.

  - 월간 정기 결제: 1~3개월 단위로 일시 중지할 수 있습니다.

  - 연간 정기 결제: 일시 중지를 지원하지 않습니다.

- 해지 예약: 사용자가 정기 결제를 이용 중이지만 취소하고 싶을 때 선택할 수 있습니다. 다음 결제일에 결제가 되지 않습니다.

- 유예, 보류: 사용자에게 결제 문제가 발생하면 다음 결제일에 결제가 되지 않습니다. 취소 예약을 할 수 없으며 즉시 "구독 해지"를 할 수 있습니다.

###### 사용자가 정기 결제를 업그레이드, 다운그레이드 또는 변경할 수 있도록 허용
사용자가 정기 결제를 업그레이드하거나 다운그레이드 하려면 구매 시 비례 배분 모드를 설정하거나 변경사항이 정기 결제 사용자에게 영향을 주는 방식을 설정할 수 있습니다.


(그림 3) : 사용자가 정기 결제 상품 버튼을 클릭하였을때 출력되는 결제 화면을 단말에서 캡쳐한 화면

다음 표에는 사용 가능한 비례 배분 모드(PurchaseFlowParams.ProrationMode)가 나와 있습니다.



비례 배분 모드에 대한 설명
- IMMEDIATE_WITH_TIME_PRORATION : 정기 결제의 교체가 즉시 이루어지며, 남은 시간은 가격 차이를 기반으로 조정되어 입금되거나 청구됩니다. (이것은 기본 
동작입니다.)
- IMMEDIATE_AND_CHARGE_PRORATED_PRICE : 정기 결제의 교체가 즉시 이루어지며, 청구 주기는 동일하게 유지됩니다. 나머지 기간에 대한 가격이 청구됩니다. (이 옵션은 업그레이드 에서만 사용할 수 있습니다.)
- IMMEDIATE_WITHOUT_PRORATION : 정기 결제의 교체가 즉시 이루어지며, 다음 결제일에 새로운 가격이 청구됩니다. 청구 주기는 동일하게 적용됩니다.
- DEFERRED : 기존 요금제가 만료되면 교체가 적용되며 새 요금이 동시에 청구됩니다.

###### 업그레이드 또는 다운그레이드
정기 결제는 구매 요청하기와 동일한 API를 사용하여 사용자에게 업그레이드 또는 다운그레이드를 제공할 수 있습니다. 다만, 정기 결제의 업그레이드 다운그레이드를 적용하기 위해선 기존 정기 결제 구매 토큰과 비례 배분 모드 값이 필수로 필요합니다.

다음 예와 같이 현재 정기 결제, 향후(업그레이드 또는 다운그레이드) 정기 결제 및 비례 배분 모드에 관한 정보를 제공해야 합니다.

```Kotlin
val subscriptionUpdateParams = SubscriptionUpdateParams.newBuilder()
      .setProrationMode(desiredProrationMode)
      .setOldPurchaseToken(oldPurchaseToken)
      .build()

val purchaseFlowParams = PurchaseFlowParams.newBuilder()
      .setProductId(newProductId)
      .setProductType(productType)
      .setProductName(productName)        // optional
      .setDeveloperPayload(devPayload)    // optional
      .setSubscriptionUpdateParams(subscriptionUdpateParams)
      .build()

purchaseClient.launchPurchaseFlow(activity, purchaseFlowParams)
```
```Java
SubscriptionUpdateParams subscriptionUpdateParams = SubscriptionUpdateParams.newBuilder()
      .setProrationMode(desiredProrationMode)
      .setOldPurchaseToken(oldPurchaseToken)
      .build();

PurchaseFlowParams purchaseFlowParams = PurchaseFlowParams.newBuilder()
      .setProductId(newProductId)
      .setProductType(productType)
      .setProductName(productName)        // optional 
      .setDeveloperPayload(devPayload)    // optional
      .setSubscriptionUpdateParams(subscriptionUdpateParams)
      .build();

purchaseClient.launchPurchaseFlow(activity, purchaseFlowParams);
```
업그레이드 또는 다운그레이드의 경우도 구매 요청하기 로직을 수행하기 때문에 응답은 PurchasesUpdatedListener에서 수신합니다. 또한 구매 내역 조회하기에서도 요청 시 응답을 받을 수 있습니다.
비례 배분 모드로 구매했을 때도 일반 구매와 동일하게 PurchaseClient.acknowledgeAsync()를 사용하여 구매 확인을 해야 합니다.

##### 구매 처리하기 
구매가 완료되면 앱에서 구매 확인 처리를 해야 합니다. 대부분의 경우 앱은 PurchasesUpdatedListener를 통해 구매 알림을 받습니다.
또는  구매 내역 조회하기에서 설명된 것처럼 앱이 PurchaseClient.queryPurchasesAsync() 함수를 호출하여 처리하는 경우가 있습니다.

다음 메서드 중 하나를 사용해 구매를 확인할 수 있습니다.

- 소모성 제품인 경우 PurchaseClient.consumeAsync()를 사용합니다.
- 소모성 제품이 아니라면 PurchaseClient.acknowledgeAsync()를 사용합니다.

###### 관리형 상품 소비하기 (consume)
관리형 상품은 소비를 하기 전까지는 재구매 할 수 없습니다.

상품을 소비하기 위해서는 consumeAsync()를 호출합니다. 또한 소비 작업 결과를 전달받으려면 ConsumeListener 인터페이스를 구현해야 합니다.

> 관리형 상품을 소비하지 않으면 영구성 형태의 상품 타입처럼 활용할 수 있으며, 구매 후 즉시 소비하면 소비성 형태의 상품으로도 활용됩니다.
또한 특정 기간 이후에 소비하면 기간제 형태의 상품으로 활용할 수 있습니다.

> 3일 이내에 구매를 확인(acknowledge) 또는 소비(consume)를 하지 않으면 사용자에게 상품이 지급되지 않았다고 판단되어 자동으로 환불됩니다.

```Kotlin
fun handlePurchase(purchase: PurchaseData) {
    // Purchase retrieved from PurchaseClient#queryPurchasesAsync
    // or your PurchasesUpdatedListener.
    val purchase: PurchaseData = ...
      
    // Verify the purchase.
    // Ensure entitlement was not already granted for this purchaseToken.
    // Grant entitlement to the user.

    val consumeParams = ConsumeParams.newBuilder()
                            .setPurchaseData(purchase)
                            .build()
                            
    purchaseClient.consumeAsync(consumeParams) { iapResult, purchaseData -> 
        // Process the result.
    }
}
```
```Java
private void handlePurchase(PurchaseData purchase) {
    // Purchase retrieved from PurchaseClient#queryPurchasesAsync or your PurchasesUpdatedListener.
    PurchaseData purchase = ...
      
    // Verify the purchase.
    // Ensure entitlement was not already granted for this purchaseToken.
    // Grant entitlement to the user.

    ConsumeParams consumeParams = ConsumeParams.newBuilder()
                                        .setPurchaseData(purchase)
                                        .build();
                                        
    purchaseClient.consumeAsync(consumeParams, new ConsumeListener() {
        @Override
        public void onConsumeResponse(IapResult iapResult, PurchaseData purchaseData) {
             // Process the result.
        }
    });
}
```
> 소비 요청이 때로 실패할 수 있으므로 보안 백엔드 서버를 확인하여 각 구매 토큰이 사용되지 않았는지 확인해야 합니다. 그래야 앱이 동일한 구매에 대해 여러 번 자격을 부여하지 않습니다. 또는 자격을 부여하기 전에 성공적인 소비 응답을 받을 때까지 기다릴 수 있습니다.

###### 구매 확인하기 (acknowledge)
비 소비성 상품에 대해 구매 확인 처리를 하려면 PurchaseClient.acknowledgeAsync() 함수를 사용합니다. 관리형 상품,  월정액 상품, 구독형 상품 모두 사용할 수 있습니다.

PurchaseData.isAcknowledged() 함수를 사용하여 구매 확인 되었는지를 판단 할 수 있습니다. 또한 구매 확인에 대한 작업 결과를 전달받으려면 AcknowledgeListener 인터페이스를 구현해야 합니다.

```Kotlin
// Purchase retrieved from PurchaseClient#queryPurchasesAsync or your PurchasesUpdatedListener.
fun handlePurchase(PurchaseData purchase) {
    if (purchase.purchaseState == PurchaseState.PURCHASED) {
        if (!purchase.isAcknowledged) {
            val acknowledgeParams = AcknowledgeParams.newBuilder()
                                        .setPurchaseData(purchase)
                                        .build()
                                        
            purchaseClient.acknowledgeAsync(acknowledgeParams) { iapResult, purchaseData ->
                 // PurchaseClient by calling the queryPurchasesAsync() method.
            }
        }
    }
}
```
```Java
// Purchase retrieved from PurchaseClient#queryPurchasesAsync or your PurchasesUpdatedListener.
private void handlePurchase(purchase: PurchaseData) {
    if (purchase.getPurchaseState() == PurchaseState.PURCHASED) {
        if (!purchase.isAcknowledged()) {
            AcknowledgeParams acknowledgeParams = AcknowledgeParams.newBuilder()
                                                        .setPurchaseData(purchase)
                                                        .build();
                                                        
            purchaseClient.acknowledgeAsync(acknowledgeParams, new AcknowledgeListener() {
                @Override
                public void onAcknowledgeResponse(IapResult iapResult, PurchaseData purchaseData) {
                    // PurchaseClient by calling the queryPurchasesAsync() method.
                }
            });
        }
    }
}
```
> AcknowledgeListener.onAcknowledgeResponse() 함수로 전달된 PurchaseData는 요청할 당시의 데이터이기 때문에 acknowledgeState 값이 변경되지 않습니다. 구매 내역 조회하기를 통해 변경된 데이터로 교체해야 합니다.

##### 구매 내역 조회하기
PurchasesUpdatedListener를 사용하여 구매를 처리하는 것만으로는 모든 구매가 처리 되었다는 것을 보장할 수 없습니다. 앱에서 구매 추적을 놓치거나 구매를 인식하지 못할 수 있는 몇 가지 시나리오는 다음과 같습니다.

아래와 같은 시나리오에서 앱은 구매 응답을 받지 못하거나 구매를 인식하지 못하는 경우가 발생할 수 있습니다. 

- 네트워크 문제: 사용자가 성공적으로 구매를 하였고, 원스토어에서 확인도 받았지만 기기가 PurchasesUpdatedListener를 통해 구매 알림을 받기 전에 네트워크 연결이 끊어졌을 경우

- 여러 기기: 한 기기에서 항목을 구입한 후 다른 기기로 전환하는 경우 

이러한 경우를 대비하여 앱의 onCreate()나 onResume()에서 PurchaseClient.queryPurchasesAsync()를 호출하여 구매가 성공적으로 처리 되었는지 확인해야 합니다.

콜백 처리는 PurchasesUpdatedListener 와 동일합니다.

```Kotlin
val queryPurchasesListener = QueryPurchasesListener { iapResult, purchases -> 
    if (iapResult.isSuccess && purchases != null) {
        for (purchase in purchases) {
            handlePurchase(purchase)
        }
    } else if (iapResult.responseCode == ResponseCode.NEED_UPDATE) {
        // PurchaseClient by calling the launchUpdateOrInstallFlow() method.
    } else if (iapResult.responseCode == ReponseCode.NEED_LOGIN) {
        // PurchaseClient by calling the launchLoginFlow() method.
    } else {
        // Handle any other error codes.
    }
}
purchaseClient.queryPurchasesAsync(ProductType.INAPP, queryPurchasesListener)
```
```Java
QueryPurchasesListener queryPurchasesListener = new QueryPurchasesListener() {
    @Override
    public void onPurchasesResponse(IapResult iapResult, list<PurchaseData> purchases) { 
        if (iapResult.isSuccess() && purchases != null) {
            for (purchase in purchases) {
                handlePurchase(purchase)
            }
        } else if (iapResult.getResponseCode() == ResponseCode.NEED_UPDATE) {
            // PurchaseClient by calling the launchUpdateOrInstallFlow() method.
        } else if (iapResult.getResponseCode() == ReponseCode.NEED_LOGIN) {
            // PurchaseClient by calling the launchLoginFlow() method.
        } else {
             // Handle any other error codes.
        }
    }
};
purchaseClient.queryPurchasesAsync(ProductType.INAPP, queryPurchasesListener);
```

##### 월정액 상품 상태 변경하기 (Deprecated)
월정액 상품은 최초 구매 후 익월 동일일에 갱신이 이루어지는 상품입니다. 월정액 상품의 상태는 PurchaseData.getRecurringState()를 통해 확인할 수 있습니다.

월정액 상품의 상태를 변경하려면 PurchaseClient.manageRecurringProductAsync()를 사용합니다.
RecurringProductParams 객체에 구매 데이터와 변경하려는 PurchaseClient.RecurringAction 값을 입력합니다. 

SDK V21 (API V7) 부터는 새로운 월정액 상품을 만들 수 없습니다. 결제 주기가 한 달인 구독형 상품을 이용하세요. 

```Kotlin
```
```Java
Copy
// Purchase retrieved from PurchaseClient#queryPurchasesAsync or your PurchasesUpdatedListener.
private void manageRecurring(PurchaseData purchase) {
    RecurringProductParams recurringParams = RecurringProductParams.newBuilder()
            .setPurchaseData(purchase)
            .setRecurringAction(RecurringAction.CANCEL | RecurringAction.REACTIVATE)
            .build();

    purchaseClient.manageRecurringProductAsync(recurringParams, new RecurringProductListener() {
        @Override
        public void onRecurringResponse(IapResult iapResult, PurchaseData purchaseData, String action) {
            // PurchaseClient by calling the queryPurchasesAsync() method.
        }
    });
}
```
> RecurringProductListener.onRecurringResponse() 함수로 전달된 PurchaseData는 요청할 당시의 데이터이기 때문에 recurringState 값이 변경되지 않습니다. 구매 내역 조회하기를 통해 변경된 데이터로 교체해야 합니다.

##### 정기 결제 관리 화면을 열기
정기 결제 중인 상품을 관리하는 화면을 띄울 수 있습니다.

매개변수로 SubscriptionsParams에 PurchaseData를 포함해서 넣으면 구매 데이터를 확인하여 해당 정기 결제 상품의 관리 화면을 실행합니다.
그러나 SubscriptionParams을 null 넣을 경우 사용자의 정기 결제 리스트 화면을 실행합니다.

다음은 정기 결제 관리 화면을 띄우는 방법을 나타내는 예제입니다.

```Kotlin
fun launchManageSubscription(@Nullable purchaseData: PurchaseData) {
    val subscriptionParams = when (purchaseData != null) {
        true -> SubscriptionParams.newBuilder()
                    .setPurchaseData(purchaseData)
                    .build()
        else -> null
    }
    purchaseClient.launchManageSubscription(mActivity, subscriptionParams)
}
```
```Java
public void launchManageSubscription(@Nullable PurchaseData purchaseData) {
    SubscriptionParams subscriptionParams = null;
    if (purchaseData != null) {
        subscriptionParams = SubscriptionParams.newBuilder()
            .setPurchaseData(purchaseData)
            .build();
    }
    purchaseClient.launchManageSubscription(mActivity, subscriptionParams);
}
```

##### 마켓 구분 코드 얻기
SDK v19 이상부터 Server API를 사용하기 위해서는 마켓 구분 코드가 필요합니다.
getStoreInfoAsync()를 통해서 마켓 구분 코드를 획득할 수 있습니다.

```Kotlin
purchaseClient.getStoreInfoAsync { iapResult, storeCode ->
    // Save storecode and use it in Server to Server API.
}
```
```Java

purchaseClient.getStoreInfoAsync(new StoreInfoListener() {
    @Override
    public void onStoreInfoResponse(IapResult iapResult, String storeCode) {
        // Save storecode and use it in Server to Server API.
    }
});
```

##### StoreEnvironment API 기능 추가
StoreEnvironment.getStoreType() API는 SDK가 탑재된 애플리케이션이 원스토어를 통해 설치되었는지를 판단하는 기능을 제공합니다.

###### Store Type 정의
해당 API는 StoreType을 반환하며, 아래 네 가지 값 중 하나를 가집니다.

StoreType, value, description을 json으로 정의
[
  {"StoreType": "StoreType.UNKNOWN", "value": 0, "description": "앱 설치 스토어 정보를 알 수 없음 (APK 직접 설치, 출처 불명 등)"},
  {"StoreType": "StoreType.ONESTORE", "value": 1, "description": "ONE Store에서 설치됨 (또는 개발자 옵션이 활성화된 경우)"},
  {"StoreType": "StoreType.VENDING", "value": 2, "description": "Google Play Store에서 설치됨"},
  {"StoreType": "StoreType.ETC", "value": 3, "description": "기타 스토어에서 설치됨"}
]

###### API 사용 방법
해당 API는 StoreEnvironment.getStoreType()을 호출하여 사용할 수 있습니다.

```Kotlin
import com.gaa.sdk.base.StoreEnvironment

val storeType = StoreEnvironment.getStoreType()

when (storeType) {
    StoreType.ONESTORE -> println("ONE Store에서 설치된 앱입니다.")
    StoreType.VENDING -> println("Google Play Store에서 설치된 앱입니다.")
    StoreType.ETC -> println("기타 스토어에서 설치된 앱입니다.")
    StoreType.UNKNOWN -> println("스토어 정보를 알 수 없습니다.")
}
```
```Java
import com.gaa.sdk.base.StoreEnvironment;

int storeType = StoreEnvironment.getStoreType();

switch (storeType) {
    case StoreType.ONESTORE:
        System.out.println("ONE Store에서 설치된 앱입니다.");
        break;
    case StoreType.VENDING:
        System.out.println("Google Play Store에서 설치된 앱입니다.");
        break;
    case StoreType.ETC:
        System.out.println("기타 스토어에서 설치된 앱입니다.");
        break;
    case StoreType.UNKNOWN:
    default:
        System.out.println("스토어 정보를 알 수 없습니다.");
        break;
}
```

###### 스토어 판단 기준
이 API는 세 가지 방법을 통해 설치된 스토어를 판별합니다.

1. 원스토어 마켓 서명을 통해 배포된 경우
원스토어의 마켓 서명을 통한 배포 여부를 확인하여, 원스토어에서 설치된 앱인지 확인합니다.

2. Installer Package Name을 기반으로 판별
원스토어의 마켓 서명을 통해 배포되지 않은 경우, PackageManager.getInstallerPackageName() API를 이용하여 앱 설치 시 사용된 스토어 정보를 확인합니다.

3. 개발자 옵션(onestore:dev_option)이 활성화된 경우
onestore:dev_option 이 설정 되어있으면 무조건 StoreType.ONESTORE로 응답합니다.

###### 활용 예시
스토어별 UI 차별화 적용

원스토어와 다른 앱 마켓에서 제공하는 결제 시스템이 다를 경우, UI를 다르게 설정할 수 있습니다.

```
if (StoreEnvironment.getStoreType() == StoreType.ONESTORE) {
    showOneStorePaymentUI()
} else {
    showDefaultPaymentUI()
}
```

스토어별 기능 차단

특정 기능을 원스토어에서만 사용하도록 설정할 수 있습니다.

```
if (StoreEnvironment.getStoreType() != StoreType.ONESTORE) {
    println("이 기능은 ONE Store에서만 사용할 수 있습니다.")
    return
}
enableOneStoreExclusiveFeature()
```
##### 원스토어 서비스 설치하기
원스토어 서비스의 버전이 낮거나 없을 경우 인앱결제를 이용할 수 없습니다. PurchaseClient.startConnection()을 통해 연결할 때 IapResult.getResponseCode()에서 확인할 수 있습니다.
RESULT_NEED_UPDATE가 발생하면 launchUpdateOrInstallFlow() 메서드를 호출해야 합니다.

```Kotlin
val activity: Activity = ...

purchaseClient.launchUpdateOrInstallFlow(activity) { iapResult ->
    if (iapResult.isSuccess) {
        // If the installation is completed successfully,
        // you should try to reconnect with the ONE store service. 
        // PurchaseClient by calling the startConnection() method.
    }
}
```

### 05. 05. 원스토어 인앱결제 레퍼런스
#### Annotations
##### PurchaseClient.ConnectionState

> public static @interface PurchaseClient.ConnectionState

 > com.gaa.sdk.iap.PurchaseClient.ConnectionState

PurchaseClient 의 연결 상태 값.

###### Constants
- CLOSED
int CLOSED
서비스의 연결이 이미 닫혔으므로 다시 사용하면 안됩니다.
Constant Value: 3

- CONNECTED
int CONNECTED
현재 서비스와 연결된 상태입니다.
Constant Value: 2

- CONNECTING
int CONNECTING
현재 서비스와 연결이 진행중인 상태입니다.
Constant Value: 1

- DISCONNECTED
int DISCONNECTED
서비스와의 연결이 끊어진 상태입니다.
Constant Value: 0

##### PurchaseClient.FeatureType
> public static @interface PurchaseClient.FeatureType

> com.gaa.sdk.iap.PurchaseClient.FeatureType

지원되는 기능 유형

###### Constants
- SUBSCRIPTIONS
String SUBSCRIPTIONS
구독 상품을 구입/관리 할 수 있습니다.
Constant Value: "subscriptions"

##### PurchaseClient.ProductType
> public static @interface PurchaseClient.ProductType

> com.gaa.sdk.iap.PurchaseClient.ProductType
지원되는 상품유형

##### Constants
- ALL
String ALL
상품상세 정보를 한번에 받기 위한 타입
Constant Value: "all"

- INAPP
String INAPP
관리형 상품 타입
Constant Value: "inapp"

- AUTO
String AUTO
월정액 상품 타입
Constant Value: "auto"

- SUBS
String SUBS
구독형 상품 타입
Constant Value: "subscription"

#### Classes
##### AcknowledgeParams

> public class AcknowledgeParams extends Object

>java.lang.Object
    ↳ com.gaa.sdk.iap.AcknowledgeParams

구매한 인앱 상품을 확인하기 위한 파라미터 입니다.

See Also:
PurchaseClient.acknowledgeAsync(AcknowledgeParams, [I]AcknowledgeListener)

Summary
Nested classes
class
[C]AcknowledgeParams.Builder
AcknowledgeParams의 인스턴스를 쉽게 만들기 위한 빌더입니다.

Public methods
getPurchaseData
[C]PurchaseData getPurchaseData()
인앱 상품의 구매 데이터를 반환합니다.
Returns: [C]PurchaseData

getDeveloperPayload
String getDeveloperPayload()
개발사에서 지정한 페이로드를 반환합니다.
Returns: String

newBuilder
[C]AcknowledgeParams.Builder newBuilder()
AcknowledgeParams의 인스턴스를 만들기 위한 빌더를 생성합니다.
Returns: [C]AcknowledgeParams.Builder

#### Interfaces

##### AcknowledgeListener

> public interface AcknowledgeListener

> com.gaa.sdk.iap.AcknowledgeListener

구매확인이 완료되면 알려주는 리스너 입니다.

###### Public methods
- onAcknowledgeResponse
> void onAcknowledgeResponse(IapResult iapResult, PurchaseData purchaseData)

구매확인 작업이 완료 되었음을 알리기 위해 호출됩니다.

Parameters:
- iapResult = 구매확인에 대한 결과
- purchaseData = 구매 데이터

### 06. 원스토어 인앱결제 서버 API (API V7)
#### 개요 
원스토어 인앱결제 서버 API란 원스토어에서 결제된 인앱 상품의 데이터를 조회하거나 결제 상태를 변경하기 위한 Open API를 말합니다.

해당 API를 사용하기 위해서는 OAuth 인증이 필요합니다.

#### HOST URL 
본 문서에서 사용되는 Host Domain은 아래와 같습니다.

원스토어 환경에 대응되는 Host Domain
- 검증(개발) : https://sbpp.onestore.net
- 상용 : https://iap-apis.onestore.net

>본 문서에서 사용하는 도메인은 모든 국가에서 공용으로 사용할 수 있는 도메인입니다. 
대한민국에서만 서비스 중인 기존 앱/게임의 도메인은 과거 도메인을 그대로 사용하시면 됩니다. 
(apis.onestore.co.kr)

#### 마켓 구분 코드 
- Request header에 마켓 구분 코드 (x-market-code)가 추가 되었습니다. 

- 원스토어 글로벌 플랫폼에 서비스 되는 경우 마켓 구분 코드를 헤더에 추가하여 요청해야 합니다. 

- 마켓 구분 코드가 없는 경우에는 원스토어(기본 값)로 호출됩니다. 

마켓 구분 코드와 서비스 대상 
- MKT_ONE : 원스토어 (기본 값)
- MKT_GLB : 원스토어 글로벌 

> 마켓 구분 코드에 따라 서버 API 응답으로 제공되는 시간의 기준이 다릅니다. 
> - MKT_ONE : UTC+09
> - MKT_GLB : UTC+00  

#### 원스토어 OAuth 
##### OAuth 개요
원스토어 서버 Open API를 연동하기 위해서는 OAuth 인증이 필요합니다.

- 원스토어 OAuth v2의 이해

  - AccessToken은 원스토어의 Server Open API를 통하여 발급받을 수 있는 값으로 원스토어에서 제공하는 Server Open API 호출시 인증값으로 사용됩니다.

  - AccessToken은 기본적으로 3600초의 유효기간이 있으며, 유효기간이 만료 되거나 600초 미만으로 남은 경우에 getAccessToken()을 호출하면 새로운 AccessToken이 발급됩니다.

    - 기존의 AccessToken도 유효기간이 끝날 때까지는 사용이 가능합니다.

    - 다수의 AccessToken이 발행되는 방식이므로 개발사의 서비스 인스턴스 별로 서로 다른 AccessToken을 취득하고 사용할 수 있는 형태가 됩니다. 


- 일반적인 연동 흐름은 다음과 같습니다.

(그림) Sequence Diagram을 설명
1. 개발사 서버 -> ONE store API Server : getAccessToken(client_id, secret)
2. 개발사 서버 <-- ONE store API Server : access token
3. 개발사 서버 -> ONE store API Server : getPurchaseDetilas(packageName, productId, purchaseToken)
4. 개발사 서버 <-- ONE store API Server : response

- AccessToken을 얻는 과정(1번)은 API 호출 시 인증 오류가 발생했을 때 호출하면 됩니다. 

원스토어의 인앱결제 서버 API의 호출을 위해서는 Authorization Bearer 스킴을 사용하며 호출 샘플은 아래와 같습니다.

```
GET /v7/apps/com.onestore.game.goindol/purchases/inapp/products/product01/SANDBOXT000120004476 HTTP/1.1
Host: sbpp.onestore.net
Authorization: Bearer 680b3621-1234-1234-1234-8adfaef561b4
```

Authorization 헤더에는 위 예제와 같이 Bearer + 공백 + AccessToken이 대소문자 구분하여 정확히 입력되어야 합니다.
Bearer의 값은 getAccessToken()을 호출하여 발급받은 AccessToken 값입니다.

잘못된 예 

```
Authorization: 680b3621-1234-1234-1234-8adfaef561b4
Authorization: bearer 680b3621-1234-1234-1234-8adfaef561b4
Authorization: Bearer <680b3621-1234-1234-1234-8adfaef561b4>
Authorization:Bearer680b3621-1234-1234-1234-8adfaef561b4
```

##### OAuth API 상세
client_id 및 client_secret 확인
Client_id와 Client_secret 값은 "라이선스 관리" 메뉴에서 확인할 수 있습니다.

##### AccessToken 발급
- URI : /v7/oauth/token
- Method: POST
- Request Parameter: Form 형식
[
  {"ParameterName": "client_id", "Description": "개발자센터에서 앱 등록 시 발급된 Client id 값", "Example": "0000042301"},
  {"ParameterName": "client_secret", "Description": "개발자센터에서 앱 등록 시 발급된 client secret 값", "Example": "vxIMAGcVz3DAx20uDBr/IDWNJAPNHFl7YruF4uxB6BI="},
  {"ParameterName": "grant_type", "Description": "고정값", "Example": "client_credentials"}
]
- Request Header : 
[
  {"ParameterName": "Content-Type", "Description": "Http 요청시 Content Type으로 반드시 application/x-www-form-urlencoded 로 설정되어야 함", "Example": "Content-Type: application/x-www-form-urlencoded"},
  {"ParameterName": "x-market-code", "Description": "마켓 구분 코드", "Example": "x-market-code: MKT_GLB"}
]

- Example 
<pre><code>
POST /v7/oauth/token HTTP/1.1
Host: apis.onestore.com
Content-Type: application/x-www-form-urlencoded;charset=UTF-8
x-market-code: MKT_GLB

grant_type=client_credentials&client_id=com.onestore.game.goindol&client_secret=vxIMAGcVz3DAx20uDBr/IDWNJAPNHFl7YruF4uxB6BI=
</code></pre>

- Response Body : JSON 형식
[
  {"ElementName": "client_id", "DataType": "String", "DataSize": 255, "Description": "OAuth 인증 client_id"},
  {"ElementName": "access_token", "DataType": "String", "DataSize": 36, "Description": "AccessToken"},
  {"ElementName": "token_type", "DataType": "String", "DataSize": 6, "Description": "bearer 방식만 제공"},
  {"ElementName": "expires_in", "DataType": "Integer", "DataSize": 10, "Description": "token 만료기한, 단위 : 초(second)"},
  {"ElementName": "scope", "DataType": "String", "DataSize": 1024, "Description": "token 사용 범위"}
]

- Example : 

```
{
    "client_id":"0000000001",
    "access_token":"680b3621-1234-1234-1234-8adfaef561b4",
    "token_type":"bearer",
    "expires_in":3010,
    "scope":"DEFAULT"
}
```

- 발급 예시 : 

```
curl -v -X POST https://sbpp.onestore.net/v7/oauth/token \
-H "Content-Type: application/x-www-form-urlencoded" \
-H "x-market-code: MKT_GLB" \
-d "grant_type=client_credentials" \
-d "client_id=0000000001" \
-d "client_secret=vxIMAGcVz3DAx20uDBr/IDWNJAPNHFl7YruF4uxB6BI="
> POST /v7/oauth/token HTTP/1.1
> Host: sbpp.onestore.net
> User-Agent: curl/7.43.0
> Accept: */*
> Content-Type: application/x-www-form-urlencoded;charset=UTF-8
> Content-Length: 35
>
* upload completely sent off: 35 out of 35 bytes
< HTTP/1.1 200 200
< Date: Wed, 02 May 2018 02:52:42 GMT
< Server: Apache
< Connection: close
< Transfer-Encoding: chunked
< Content-Type: application/json;charset=UTF-8
<
* Closing connection 0
{"client_id":"0000000001","access_token":"680b3621-1234-1234-1234-8adfaef561b4","token_type":"bearer","expires_in":3600,"scope":"DEFAULT"}
```

검증(개발) 환경과 상용환경의 AccessToken은 독립적으로 관리되므로, 환경 별로 AccessToken을 분리하여 관리하셔야 합니다.

#### 서버 API 상세 
> 2025년 3월 20일 개발자센터 개편의 영향으로 서버 API 요청 시 이용되던 packageName은 clientId로 변경되었습니다. 
> - 3월 20일 이전 등록된 앱의 경우 클라이언트 ID 와 패키지명이 동일하므로, 변경에 대한 영향도가 없습니다. 
> - 3월 20일 이후 등록된 앱의 경우 클라이언트 ID 가 PID와 동일하므로 참고 부탁 드립니다. 

##### getPurchaseDetails (구매상품 상세조회)
- Desc : 구매한 원스토어 관리형 상품의 상세정보를 조회합니다. 구매완료 시 전달받은 원스토어 purchaseToken(구매 토큰)을 이용하여 조회하여야 합니다. 월정액 상품의 purchaseToken으로 조회 시, 조회 실패(NoSuchData) 응답이 전달됩니다.

- URI : /v7/apps/{clientId}/purchases/inapp/products/{productId}/{purchaseToken}
- Method : GET
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  - String productId : 상품 ID (Data Size : 150)
  - String purchaseToken : 구매 토큰 (Data Size : 20)
- Request Header: 
[
  {"Parameter Name": "Authorization", "DataType": "String", "Required": true, "Description": "Access Token API를 통해 발급받은 access_token"}, 
  {"Parameter Name": "Content-Type", "DataType": "String", "Required": true, "Description": "application/json"}, 
  {"Parameter Name": "x-market-code", "DataType": "String", "Required": false, "Description": "마켓 구분 코드"}
]

- Example 

<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Request Body : 없음

- Response Body : JSON 형식
[
  {"ElementName": "consumptionState", "DataType": "Integer", "DataSize": 1, "Description": "구매한 상품의 소비상태 (0: 미소비, 1: 소비)"},
  {"ElementName": "developerPayload", "DataType": "String", "DataSize": 200, "Description": "개발사가 제공한 결제 고유 식별자"},
  {"ElementName": "purchaseState", "DataType": "Integer", "DataSize": 1, "Description": "구매상태 ( 0: 구매완료, 1: 취소완료)"},
  {"ElementName": "purchaseTime", "DataType": "Long", "DataSize": 13, "Description": "구매시간 (ms)"},
  {"ElementName": "purchaseId", "DataType": "String", "DataSize": 20, "Description": "구매 ID"},
  {"ElementName": "acknowledgeState", "DataType": "Integer", "DataSize": 1, "Description": "구매확인 상태( 0: Not Acknowledged, 1: Acknowledged)"},
  {"ElementName": "quantity", "DataType": "Integer", "DataSize": 2, "Description": "구매 수량"}
]

- Example:
<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "consumptionState": 0,
    "developerPayload": "developerPayload",
    "purchaseState": 0,
    "purchaseTime": 1345678900000,
    "purchaseId": "17070421461015116878",
    "acknowledgeState": 0,
	"quantity": 2
}
</code></pre>

##### getRecurringPurchaseDetails (월정액 상품 구매 상세조회) 
- Desc : 구매한 원스토어 월정액 상품의 자동결제 상태와 마지막 구매상태의 상세정보를 조회합니다. 구매완료 시 전달받은 원스토어 purchaseToken(구매 토큰)을 이용하여 조회하여야 합니다. 관리형 상품의 purchaseToken으로 조회 시, 조회 실패(NoSuchData) 응답이 전달됩니다.
- URI : /v7/apps/{clientId}/purchases/auto/products/{productId}/{purchaseToken}
- Method : GET
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  - String productId : 상품 ID (Data Size : 150)
  - String purchaseToken : 구매 토큰 (Data Size : 20)
- Request Header: 
[
  {"ParameterName": "Authorization", "DataType": "String", "Required": true, "Description": "Access Token API를 통해 발급받은 access_token"}, 
  {"ParameterName": "Content-Type", "DataType": "String", "Required": true, "Description": "application/json"}, 
  {"ParameterName": "x-market-code", "DataType": "String", "Required": false, "Description": "마켓 구분 코드"}
]

- Example 
<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Request Body : 없음
- Response Body : JSON 형식
[
  {"ElementName": "startTime", "DataType": "Long", "DataSize": 13, "Description": "구매한 상품의 이용시작시간(ms)"},
  {"ElementName": "expiryTime", "DataType": "Long", "DataSize": 13, "Description": "구매한 상품의 이용종료시간(ms)"},
  {"ElementName": "nextPaymentTime", "DataType": "Long", "DataSize": 13, "Description": "다음 자동결제시간"},
  {"ElementName": "autoRenewing", "DataType": "Boolean", "DataSize": 0, "Description": "이용종료시간(expiryTime)가 지난 후 자동결제 여부"},
  {"ElementName": "cancelReason", "DataType": "Integer", "DataSize": 1, "Description": "해지사유 (0 : 고객요청에 의한 해지, 1 : 기타 시스템 처리로 인한 해지 )"},
  {"ElementName": "cancelledTime", "DataType": "Long", "DataSize": 13, "Description": "해지시간(ms)"},
  {"ElementName": "acknowledgeState", "DataType": "Integer", "DataSize": 1, "Description": "월정액 상품의 구매확인 상태( 0: Not Acknowledged, 1: Acknowledged)"},
  {"ElementName": "lastPurchaseId", "DataType": "String", "DataSize": 20, "Description": "마지막에 자동결제된 구매ID"},
  {"ElementName": "lastPurchaseState", "DataType": "Integer", "DataSize": 1, "Description": "마지막에 자동결제된 구매상태 (0 : 구매완료, 1 : 취소완료)"}
]

- Example : 
<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "startTime": 1345678900000,
    "expiryTime": 1345678999999,
    "nextPaymentTime": 1345688000000,
    "autoRenewing": true,
    "cancelReason": 1,
    "cancelledTime": 1345679000000,
    "acknowledgeState": 0,
    "lastPurchaseId":"15081718460701027851",
    "lastPurchaseState": 0
}
</code></pre>
고객이 월정액 상품에 대하여 컨텐츠 사용 권한이 있는지 아래의 조건으로 판단이 가능합니다.

- 현재시간이 expiryTime(구매한 상품의 이용종료시간)보다 작거나 같고, lastPurchaseState(마지막에 자동결제된 구매상태)가 0(구매완료) 상태인경우
Ex) expiryTime >= 현재시간 AND lastPurchaseState == 0

##### acknowledgePurchase (구매상품 확인) 
- Desc : 구매한 관리형 또는 월정액 상품을 구매확인 상태로 변경한다.
- URI : /v7/apps/{clientId}/purchases/all/products/{productId}/{purchaseToken}/acknowledge
- Method : POST
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  - String productId : 상품 ID (Data Size : 150)
  - String purchaseToken : 구매 토큰 (Data Size : 20)
- Request Header: 
[
  {"ParameterName": "Authorization", "DataType": "String", "Required": true, "Description": "Access Token API를 통해 발급받은 access_token"}, 
  {"ParameterName": "Content-Type", "DataType": "String", "Required": true, "Description": "application/json"}, 
  {"ParameterName": "x-market-code", "DataType": "String", "Required": false, "Description": "마켓 구분 코드"}
]

- Example 
<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Request Body : JSON 형식
[
  {"ParameterName": "developerPayload", "DataType": "String", "Required": false, "Description": ""}
]

- Example :
<pre><code>
{
    "developerPayload": "your payload"
}
</code></pre>

- Response Body : JSON 형식
API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.
[
  {"ElementName": "code", "DataType": "String", "DataSize": 256, "Description": "응답코드"},
  {"ElementName": "message", "DataType": "String", "DataSize": 256, "Description": "응답 메시지"},
  {"ElementName": "result", "DataType": "Object", "DataSize": 13, "Description": "다음 자동결제시간"}
]

<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}
</code></pre>

원스토어는 3일 이내에 acknowledgePurchase API가 호출되지 않은 구매 건을 자동으로 취소합니다.

이에 따라 개발사는 반드시 해당 API를 호출해야 하며, SDK API 또는 서버 API를 통해 처리할 수 있습니다.

단, consumePurchase API가 호출된 구매 건은 acknowledge가 되었다고 판단하고 구매 취소를 하지 않습니다.

##### consumePurchase (구매상품 소비)
- Desc : 구매한 관리형 인앱 상품을 소비 상태로 변경한다.
- URI : /v7/apps/{clientId}/purchases/inapp/products/{productId}/{purchaseToken}/consume
- Method : POST
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  - String productId : 상품 ID (Data Size : 150)
  - String purchaseToken : 구매 토큰 (Data Size : 20)

- Request Header: 
[
  {"ParameterName": "Authorization", "DataType": "String", "Required": true, "Description": "Access Token API를 통해 발급받은 access_token"}, 
  {"ParameterName": "Content-Type", "DataType": "String", "Required": true, "Description": "application/json"}, 
  {"ParameterName": "x-market-code", "DataType": "String", "Required": false, "Description": "마켓 구분 코드"}
]

- Example 
<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Request Body : JSON 형식
[
  {"ParameterName": "developerPayload", "DataType": "String", "Required": false, "Description": ""}
]

- Example :
<pre><code>
{
    "developerPayload": "your payload"
}
</code></pre>

- Response Body : JSON 형식
API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.
[
  {"ElementName": "code", "DataType": "String", "DataSize": 256, "Description": "응답코드"},
  {"ElementName": "message", "DataType": "String", "DataSize": 256, "Description": "응답 메시지"},
  {"ElementName": "result", "DataType": "Object", "DataSize": 0, "Description": ""}
]

<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}
</code></pre>

##### cancelRecurringPurchase (자동결제 해지요청) 
- Desc : 월정액 상품의 자동결제 해지를 요청한다. 구독형 상품의 purchaseToken이용 시, 조회 실패(NoSuchData) 응답이 전달됩니다.
- URI : /v7/apps/{clientId}/purchases/auto/products/{productId}/{purchaseToken}/cancel
- Method : POST
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  - String productId : 상품 ID (Data Size : 150)
  - String purchaseToken : 구매 토큰 (Data Size : 20)

- Request Header: 
[
  {"ParameterName": "Authorization", "DataType": "String", "Required": true, "Description": "Access Token API를 통해 발급받은 access_token"}, 
  {"ParameterName": "Content-Type", "DataType": "String", "Required": true, "Description": "application/json"}, 
  {"ParameterName": "x-market-code", "DataType": "String", "Required": false, "Description": "마켓 구분 코드"}
]

- Example 
<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Response Body : JSON 형식
API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.
[
  {"ElementName": "code", "DataType": "String", "DataSize": 256, "Description": "응답코드"},
  {"ElementName": "message", "DataType": "String", "DataSize": 256, "Description": "응답 메시지"},
  {"ElementName": "result", "DataType": "Object", "DataSize": 0, "Description": ""}
]

<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}
</code></pre>


##### reactiveRecurringPurchase (자동결제 해지 취소요청) 
- Desc : 월정액 상품의 자동결제 해지요청을 취소한다. 구독형 상품의 purchaseToken이용 시, 조회 실패(NoSuchData) 응답이 전달됩니다.
- URI : /v7/apps/{clientId}/purchases/auto/products/{productId}/{purchaseToken}/reactivate
- Method : POST
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  - String productId : 상품 ID (Data Size : 150)
  - String purchaseToken : 구매 토큰 (Data Size : 20)

- Request Header: 
[
  {"ParameterName": "Authorization", "DataType": "String", "Required": true, "Description": "Access Token API를 통해 발급받은 access_token"}, 
  {"ParameterName": "Content-Type", "DataType": "String", "Required": true, "Description": "application/json"}, 
  {"ParameterName": "x-market-code", "DataType": "String", "Required": false, "Description": "마켓 구분 코드"}
]

- Example 
<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Response Body : JSON 형식
API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.
[
  {"ElementName": "code", "DataType": "String", "DataSize": 256, "Description": "응답코드"},
  {"ElementName": "message", "DataType": "String", "DataSize": 256, "Description": "응답 메시지"},
  {"ElementName": "result", "DataType": "Object", "DataSize": 0, "Description": ""}
]

<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}
</code></pre>

##### getVoidedPurchases (구매취소내역 조회) 
- Desc : 구매취소내역을 조회한다.
- URI : /v7/apps/{clientId}/voided-purchases
- Method : GET
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
- Request Parameter (Optional) : Query String 형식
  - String continuationKey : 구매취소 건이 많을 경우 원스토어 서버에서 이 값을 반환합니다. 
응답에 continuationKey가 있을 경우, getVoidedPurchases를 다시 호출하면서 continuationKey를 전달하면 추가 구매취소 내역을 전달받을 수 있습니다. (Data Size : 41)
  - String startTime : 구매취소일시 검색 시작시간 (milliseconds). 
현재시간기준 과거 1개월까지만 설정가능하며 startTime 단독으로 사용할 경우 endTime은 startTime기준 미래 1개월로 설정됩니다. (Data Size : 13)
  - String endTime :  구매취소일시 검색 종료시간 (milliseconds). 
현재시간보다 클 수 없으며 endTime 단독으로 사용할 경우 startTime은 endTime기준 과거 1개월로 설정됩니다. (Data Size : 13)
  - unsigned integer maxResults : 최대조회건수 default 100 (Data Size : 3)
- Request Header: 
[
  {"ParameterName": "Authorization", "DataType": "String", "Required": true, "Description": "Access Token API를 통해 발급받은 access_token"}, 
  {"ParameterName": "Content-Type", "DataType": "String", "Required": true, "Description": "application/json"}, 
  {"ParameterName": "x-market-code", "DataType": "String", "Required": false, "Description": "마켓 구분 코드"}
]

- Example 
<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Response Body
  - continuationKey : 구매취소 건이 최대조회건수보다 많을 경우 반환. 이후 요청시 전달받은 continuationKey 를 셋팅하여 호출하면 이후 내역을 조회 할 수 있음
  - voidedPurchaseList
    - purchaseId : 구매 ID
    - purchaseTime : 구매시간(ms)
    - voidedTime : 구매취소시간(ms)
    - purchaseToken : 구매 토큰
    - marketCode : 마켓 구분 코드

- Example : 
<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "continuationKey": "continuationKey",
    "voidedPurchaseList ": [
        {
            "purchaseId": "19062709124410111299",
            "purchaseTime": 1345678900000,
            "voidedTime": 1345688900000,
            "purchaseToken": "19062709124410111299",
            "marketCode": "MKT_ONE"
        },
        {
            "purchaseId": "19062709124410111300",
            "purchaseTime": 1345679900000,
            "voidedTime": 1345878900000,
            "purchaseToken": "19062709124410111299",
            "marketCode": "MKT_ONE"
        }
    ]
}
</code></pre>

##### getSubscriptionDetail (구독 상세 조회) 
- Desc : 구독 상세 내역을 조회한다.
- URI : /v7/apps/{clientId}/purchases/subscription/products/{productId}/{purchaseToken}
- Method : GET
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  - String productId : 상품 ID (Data Size : 150)
  - String purchaseToken : 구매 토큰 (Data Size : 20)

- Request Header: 
  - Authorization: Access Token API를 통해 발급받은 access_token
  - Content-Type: application/json
  - x-market-code: 마켓 구분 코드

- Example 

<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Request Body : 없음
- Response Body
  - acknowledgementState : 구매확인 상태( 0: Not Acknowledged, 1: Acknowledged)
  - developerPayload : 개발사가 제공한 결제 고유 식별자
  - autoRenewing : 다음결제 갱신여부(구독해지예약,해지,만료 : false, 나머지 : true)
  - paymentState : 구독결제 수신여부(null: 만료 상태, 0: 결제가 완료 되지 않은 상태, 1: 결제요청 후 결제완료가 된 상태, 2: 무료기간으로 처리된 상태, 3: 상품 업/다운그레이드 시 DEFERRED옵션으로 처리된 상태)
  - priceAmount : 구매금액
  - priceAmountMicros : 구매금액 * 100만
  - nextPriceAmount : 다음구매금액
  - nextPriceAmountMicros : 다음구매금액 * 100만
  - nextPaymentTimeMillis : 다음구매시간(ms)
  - priceCurrencyCode : 통화코드(KRW 고정)
  - countryCode : 국가코드(KR 고정)
  - startTimeMillis : 구독 시작(첫 결제)시간(ms)
  - expiryTimeMillis : 구독 만료시간(ms)
  - pauseStartTimeMillis : 일시정지 시작일시(ms)-일시정지예약/일시정지 시에만 존재
  - pauseEndTimeMillis : 일시정지 시작일시(ms)-일시정지예약/일시정지 시에만 존재
  - autoResumeTimeMillis : 일시정지 후 재구독 시간(정상 구독 중: null, 일시정지 예약/확정/일시정지중: 다음결제일 + 일시정지일자)
  - linkedPurchaseToken : 구독상품 변경 시 이전 purchaseToken, 변경한 적이 없으면 null
  - lastPurchaseId : 마지막 구매ID
  - cancelledTimeMillis : 구독 해지시간(ms)
  - cancelReason : 구독 해지사유 (0 : 고객요청, 1 : 기타)
  - promotionPrice : 프로모션 금액 정보
    - promotionPrice : 프로모션 금액
    - promotionPriceMicros : 프로모션 금액 정보 * 100만
    - promotionPeriod : 프로모션 적용 회차
  - priceChange : 가격변동정보
    - seq : 가격변동 시퀀스 
    - previousPrice : 이전 가격
    - previousPriceMicros : 이전 가격 * 100만
    - newPrice : 신규 가격
    - newPriceMicros : 신규 가격 * 100만
    - agreement : 가격 변동 동의 여부
    - agreementDueDateTimeMillis : 가격 변동 동의 만료 시간(ms)(정책 부연 설명) Value = 가격변경일 +7+30일, 사용자는 동의만료일 이후 첫 자동결제 시점까지 동의가능함.

- Example : 
<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "acknowledgementState":1,
    "autoRenewing":true,
    "paymentState":1,
    "lastPurchaseId": "20202394820938409234"
    "linkedPurchaseToken":null,
    "priceAmount":"100",
    "priceAmountMicros":100000000,
    "nextPriceAmount":"150",
    "nextPriceAmountMicros":150000000,
    "nextPaymentTimeMillis": 1623337199000,
    "priceCurrencyCode":"KRW",
    "countryCode":"KR",
    "startTimeMillis":1623337199000,
    "expiryTimeMillis":1625929199000,
    "pauseStartTimeMillis":1625929199000,
    "pauseEndTimeMillis":1625929199000,
    "autoResumeTimeMillis":1625878800000,
    "cancelledTimeMillis":1625929199000,
    "cancelReason":0,
    "promotionPrice":{
        "promotionPrice":"100",
        "promotionPriceMicros":100000000,
        "promotionPeriod":30
    },
    "priceChange":{
        "seq": 1,
        "previousPrice":"100",
        "previousPriceMicros":100000000,
        "newPrice":"500",
        "newPriceMicros":500000000,
        "applyTimeMillis":1625670000000,
        "agreement":false,
        "agreementDueDateTimeMillis": 1345678920000
    }
}
</code></pre>

##### cancelSubscription (구독결제 해지요청) 
- Desc : 구독형 상품의 자동결제 해지를 요청한다. 단, 요청 시점에 구독상태가 일시중지, 결제유예, 계정보류일 경우 즉시 해지를 요청한다.
- URI : /v7/apps/{clientId}/purchases/subscription/products/{productId}/{purchaseToken}/cancel
- Method : POST
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  - String productId : 상품 ID (Data Size : 150)
  - String purchaseToken : 구매 토큰 (Data Size : 20)

- Request Header: 
[
  {"ParameterName": "Authorization", "DataType": "String", "Required": true, "Description": "Access Token API를 통해 발급받은 access_token"}, 
  {"ParameterName": "Content-Type", "DataType": "String", "Required": true, "Description": "application/json"}, 
  {"ParameterName": "x-market-code", "DataType": "String", "Required": false, "Description": "마켓 구분 코드"}
]

- Example 
<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Response Body : JSON 형식
API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.
[
  {"ElementName": "code", "DataType": "String", "DataSize": 256, "Description": "응답코드"},
  {"ElementName": "message", "DataType": "String", "DataSize": 256, "Description": "응답 메시지"},
  {"ElementName": "result", "DataType": "Object", "DataSize": 0, "Description": ""}
]

<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}
</code></pre>

##### reactivateSubscription (구독결제 해지 취소요청) 
- Desc : 구독형 상품의 자동결제 해지요청을 취소한다. 단, 즉시해지된 경우 해지요청을 취소할 수 없다.
- URI : /v7/apps/{clientId}/purchases/subscription/products/{productId}/{purchaseToken}/reactivate
- Method : POST
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  - String productId : 상품 ID (Data Size : 150)
  - String purchaseToken : 구매 토큰 (Data Size : 20)

- Request Header: 
[
  {"ParameterName": "Authorization", "DataType": "String", "Required": true, "Description": "Access Token API를 통해 발급받은 access_token"}, 
  {"ParameterName": "Content-Type", "DataType": "String", "Required": true, "Description": "application/json"}, 
  {"ParameterName": "x-market-code", "DataType": "String", "Required": false, "Description": "마켓 구분 코드"}
]

- Example 
<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Response Body : JSON 형식
API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.
[
  {"ElementName": "code", "DataType": "String", "DataSize": 256, "Description": "응답코드"},
  {"ElementName": "message", "DataType": "String", "DataSize": 256, "Description": "응답 메시지"},
  {"ElementName": "result", "DataType": "Object", "DataSize": 0, "Description": ""}
]

<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}
</code></pre>

##### deferSubscription (구독결제 연기요청) 
- Desc : 구독형 상품의 다음 결제일을 연기한다.
- URI : /v7/apps/{clientId}/purchases/subscription/products/{productId}/{purchaseToken}/defer
- Method : POST
- Request Parameter : Path Variable 형식
  - String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  - String productId : 상품 ID (Data Size : 150)
  - String purchaseToken : 구매 토큰 (Data Size : 20)

- Request Header: 
[
  {"ParameterName": "Authorization", "DataType": "String", "Required": true, "Description": "Access Token API를 통해 발급받은 access_token"}, 
  {"ParameterName": "Content-Type", "DataType": "String", "Required": true, "Description": "application/json"}, 
  {"ParameterName": "x-market-code", "DataType": "String", "Required": false, "Description": "마켓 구분 코드"}
]

- Example 
<pre><code>
Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
Request.setHeader("Content-Type", "application/json");
Request.setHeader("x-market-code", "MKT_GLB");
</code></pre>

- Request Body : JSON 형식
[
  {"ParameterName": "deferPeriod", "DataType": "Integer", "Required": true, "Description": "연기 기간, 상용: 일 단위(1~365일), 샌드박스: 분 단위"}
]

- Example : 
<pre><code>
Content-type: application/json;charset=UTF-8
{
	"deferPeriod" : 3
}
</code></pre>

- Response Body : JSON 형식
API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.
[
  {"ElementName": "code", "DataType": "String", "DataSize": 256, "Description": "응답코드"},
  {"ElementName": "message", "DataType": "String", "DataSize": 256, "Description": "응답 메시지"},
  {"ElementName": "result", "DataType": "Object", "DataSize": 0, "Description": ""}
]

<pre><code>
HTTP/1.1 200 OK
Content-type: application/json;charset=UTF-8
{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}
</code></pre>

###### 표준응답규격
- 표준응답코드
[
  {"Code": "AccessBlocked", "Message": "The request was blocked.", "Description": "요청이 차단되었습니다.", "HTTP Status Code": "403 - Fobidden", "TargetAPI": "공통"},
  {"Code": "AccessTokenExpired", "Message": "Access token has expired..", "Description": "Access 토큰이 만료되었습니다.", "HTTP Status Code": "401 - Unauthorized", "TargetAPI": "공통"},
  {"Code": "BadRequest", "Message": "The request are invalid.", "Description": "잘못된 요청입니다.", "HTTP Status Code": "400 - Bad Request", "TargetAPI": "공통"},
  {"Code": "DeveloperPayloadNotMatch", "Message": "The request developerPayload does not match the value passed in the purchase request.", "Description": "구매요청 시 전달된 developerPayload값과 일치하지 않습니다.", "HTTP Status Code": "500 - Internal Server Error", "TargetAPI": "공통"},
  {"Code": "InternalError", "Message": "An undefined error has occurred.", "Description": "정의되지 않은 오류가 발생하였습니다.", "HTTP Status Code": "403 - Fobidden", "TargetAPI": "공통"},
  {"Code": "InvalidAccessToken", "Message": "Access token is invalid.", "Description": "Access 토큰이 유효하지 않습니다.", "HTTP Status Code": "401 - Unauthorized", "TargetAPI": "공통"},
  {"Code": "InvalidAuthorizationHeader", "Message": "Authorization header is invalid.", "Description": "Authorization 헤더의 값이 유효하지 않습니다.", "HTTP Status Code": "400 - Bad Request", "TargetAPI": "공통"},
  {"Code": "InvalidConsumeState", "Message": "The purchase consumption status cannot be changed or has already been changed.", "Description": "소비상태 변경이 불가하거나, 이미 변경완료 되었습니다.", "HTTP Status Code": "409 - Conflict", "TargetAPI": "consumePurchase"},
  {"Code": "InvalidContentType", "Message": "The request content-type is invalid.", "Description": "잘못된 Content Type 입니다.", "HTTP Status Code": 415 - Unsupported Media Type", "TargetAPI": "공통"},
  {"Code": "InvalidPurchaseState", "Message": "Purchase history does not exist or is not completed.", "Description": "구매내역이 존재하지 않거나, 구매완료 상태가 아닙니다.", "HTTP Status Code": "409 - Conflict", "TargetAPI": "acknowledgePurchase, consumePurchase"},
  {"Code": "InvalidRequest", "Message": "Request parameters are invalid. [ field1, field2, ... ]", "Description": "입력값이 유효하지 않습니다. [ field1, field2, ... ]", "HTTP Status Code": "400 - Bad Request", "TargetAPI": "공통"},
  {"Code": "MethodNotAllowed", "Message": "HTTP method not supported.", "Description": "지원하지 않는 HTTP Method 입니다.", "HTTP Status Code": "405 - Method Not Allowed", "TargetAPI": "공통"},
  {"Code": "NoSuchData", "Message": "The requested data could not be found.", "Description": "조회된 결과값이 존재하지 않습니다.", "HTTP Status Code": "404 - Not Found", "TargetAPI": "getPurchaseDetails, getRecurringPurchaseDetails"},
  {"Code": "RequiredValueNotExist", "Message": "Request parameters are required. [ field1, field2, ... ]", "Description": "필수값이 존재하지 않습니다. [ field1, field2, ... ]", "HTTP Status Code": "400 - Bad Request", "TargetAPI": "공통"},
  {"Code": "ResourceNotFound", "Message": "The requested resource could not be found.", "Description": "요청한 자원이 존재하지 않습니다.", "HTTP Status Code": "404 - Not Found", "TargetAPI": "공통"},
  {"Code": "ServiceMaintenance", "Message": "System maintenance is in progress.", "Description": "서비스 점검중입니다.", "HTTP Status Code": "503 - Service Temporarily Unavailable", "TargetAPI": "공통"},
  {"Code": "Success", "Message": "The request has been completed successfully.", "Description": "정상처리 되었습니다.", "HTTP Status Code": "200 - Success", "TargetAPI": "acknowledgePurchase, consumePurchase, cancelRecurringPurchase, reactivateRecurringPurchase"},
  {"Code": "UnauthorizedAccess", "Message": "Not authorized to this API.", "Description": "해당 API에 접근권한이 없습니다.", "HTTP Status Code": "403 - Fobidden", "TargetAPI": "공통"},
]

###### 표준오류 응답규격
서버 API에서는 정상적인 응답 외에 오류발생 시, 아래의 Example과 같은 형식의 표준오류응답을 리턴합니다. 아래 내용을 참고하시기 바랍니다.

- Response Body : JSON 형식
  - code : 응답 코드
  - message : 응답 메시지
  - error : Object

- Example

<pre><code>
HTTP/1.1 400 Bad Request
Content-type: application/json;charset=UTF-8
{
    "error" : {
        "code" : "NoSuchData",
        "message" : "The requested data could not be found."
    }
}
</code></pre>

##### 공통 코드 
###### 상품타입 코드 
[
  {"code": "inapp", "name": "관리형 상품", "description": "소비성/영구성/기간제 상품"}, 
  {"code": "auto", "name": "월정액 상품", "description": "월 자동결제 상품"}, 
  {"code": "subscription", "name": "구독형 상품", "description": "구독형(자동결제) 상품"}, 
  {"code": "all", "name": "전체 상품", "description": "관리형 상품 + 월정액 상품"}
]

### 07. PNS (Payment Notification Service) 이용하기
#### 개요 
원스토어는 개발자를 위해 두 가지 Payment Notification Service를 제공합니다.
* PNS는 Payment Notification Service의 약자입니다.
* PNS는 모바일의 네트워크 연결 불안정성을 보완하기 위해 개발사가 지정한 서버로 원스토어의 서버가 개별 사용자의 결제 상태(결제 완료, 결제 취소)를 메시지로 전송하여 결제 트랜젝션의 상태를 손실없이 알려주기 위한 용도의 기능입니다.
* 즉, 개발사가 지정한 서버에서 원스토어가 정의한 규칙에 맞추어 API를 구현하면 해당 API를 원스토어의 결제 담당 서버에서 호출하는 형태입니다.
* Server to Server, 즉 서버간에 데이터를 전송한다고 할지라도 네트워크 문제로 메세지 전송 실패가 발생하기 때문에 200 OK로 응답을 인지하지 못할 경우 반복하여 메시지가 전송될 수 있습니다. 개발사의 서버는 메시지를 수신후 정의된 응답을 하여야 원스토어는 개발사 서버가 정상적으로 메시지를 전달 받았음을 인지합니다.
결제 트렌젝션의 유형에 따라 아래와 같은 유형의 메세지가 전송됩니다.

- 인앱상품 결제 또는 결제취소가 발생하면 원스토어가 개발사 서버로 알림을 전송하는 PNS(Payment Notification Service) 

- 구독 상태가 변경되면 개발사 서버로 알림을 전송하는 SNS (Subscription Notifacation Service) 

> Notification은 발송/수신 서버의 상태에 따라 지연 또는 유실될 수 있으므로, notification 수신을 기준으로 상품(서비스)을 제공하는 것은 권장하지 않습니다. 
정상적인 결제 건인지 Server to Server로 확인하기를 원하신다면 PNS notification을 이용하는 대신, 관련 서버 API로 조회하는 것을 권장합니다.
원스토어는 검증 및 모니터링 목적으로 결제 테스트를 진행할 수 있으며, 해당 테스트 건들도 결제/결제취소 시 동일하게 notification이 발송됩니다. 원스토어가 진행한 결제 테스트 내역은 주기적으로 원스토어에서 자체 취소 처리합니다.

#### PNS 수신 서버 URL 설정
PNS를 수신 받을 개발사 서버의 URL은 '개발자센터 > Apps > 상품 선택 > In-App정보' 메뉴에서 'PNS 관리' 버튼을 클릭하면 설정할 수 있습니다.
URL은 Sandbox(개발용) 결제환경 및 상용(상용테스트 포함) 결제환경을 각각 설정할 수 있으며, 개발용/상용 서버가 동일할 경우 동일한 URL을 입력하시면 됩니다.

#### PNS 상세
> 2025년 3월 20일 개발자센터 개편의 영향으로 PNS 3.1.0 버전이 추가 되었습니다. 
> - packageName 파라미터가 clientId로 변경 되었으며, 3.0.0 이하 버전에서는 변경 사항이 없습니다.

##### Payment Notification 메시지 발송 규격 (원스토어가 개발사 서버 API를 호출할 때의 규격) 
야래는 사용자가 결제시 원스토어의 결제 시스템에서 개발사의 서버로 발송하는 메세지에 포함된 요소들이다. 해당 요소들은 HTTP/JSON 형태의 메세지로 개발사로 전달된다.
- URI : 개발자 센터에서 설정한 Payment Notification URL
- Method : POST
- Request Parameters : N/A
- Request Header : 
<pre><code>
Content-Type: application/json
</code></pre>
- Request Body : JSON 형식
  - msgVersion : 메시지 버전(개발(Sandbox) : 3.1.0D, 상용(상용테스트) : 3.1.0)
  - clientId : 앱의 클라이언트 ID
  - productId : 인앱상품의 상품 ID
  - messageType : SINGLE_PAYMENT_TRANSACTION 고정
  - purchaseId : 구매 ID
  - developerPayload : 구매건을 식별하기 위해 개발사에서 관리하는 식별자
  - purchaseTimeMillis : 원스토어 결제 시스템에서 결제가 완료된 시간(ms)
  - purcahseState : COMPLETED : 결제완료 / CANCELED : 취소
  - price : 결제 금액
  - priceCurrencyCode : 결제 금액 통화코드(KRW, USD, ...)
  - productName : 구매요청 시 개발사가 customized 인앱상품 제목을 설정한 경우 전달
  - paymentTypeList : 결제 정보 목록
    - paymentMethod : 결제 수단 (상세 내용은 아래 paymentMethod 정의 참고)
    - amount : 결제 수단 별 금액
  - billingKey : 확장 기능용 결제 키
  - isTestMdn : 시험폰 여부(true : 시험폰, false : 시험폰 아님)
  - purchaseToken : 구매토큰
  - environment : 결제환경(개발(샌드박스) : SANDBOX, 상용 :COMMERCIAL)
  - marketCode : 마켓 구분코드 ( MKT_ONE : 원스토어, MKT_STM : 스톰 )
  - signature : 본 메시지에 대한 signature

아래는 실제 전송되는 PNS 결제 완료의 Json message의 예시이다.
```json
{
	"msgVersion" : "3.1.0"
	"clientId":"0000000001",
	"productId":"0900001234",
	"messageType":"SINGLE_PAYMENT_TRANSACTION",
	"purchaseId":"SANDBOX3000000004564",
	"developerPayload":"OS_000211234",
	"purchaseTimeMillis":24431212233,
	"purchaseState":"COMPLETED",
	"price":"10000",
	"priceCurrencyCode":"KRW"
	"productName":"GOLD100(+20)"
	"paymentTypeList":[
		{
			"paymentMethod":"DCB",
			"amount":"3000"
		},
		{
			"paymentMethod":"ONESTORECASH",
			"amount":"7000"
		}
	],
	"billingKey" : "36FED4C6E4AC9E29ADAF356057DB98B5CB92126B1D52E8757701E3A261AF49CCFBFC49F5FEF6E277A7A10E9076B523D839E9D84CE9225498155C5065529E22F5",
	"isTestMdn" : true,
	"purchaseToken" : "TOKEN...",
	"environment" : "SANDBOX",
	"marketCode" : "MKT_ONE"
	"signature" "SIGNATURE..."
}
```

##### paymentMethod(원스토어 결제수단) 정의
아래는 paymentMethod: 결제수단 명칭(설명) 정의
- DCB : 휴대폰결제(통신사 요금청구서에 '정보이용료' 항목으로 청구)
- PHONEBILL : 휴대폰 소액결제(통신사 요금청구서에 '소액결제' 항목으로 청구)
- ONEPAY : ONE pay(원스토어가 제공하는 간편결제)
- CREDITCARD : 신용카드(일반 신용카드 결제)
- 11PAY : 11Pay(11번가에서 제공하는 신용카드 간편결제)
- NAVERPAY : N pay(네이버에서 제공하는 네이버페이 결제)
- CULTURELAND : 컬쳐캐쉬(한국문화진흥에서 제공한는 컬쳐캐쉬 결제)
- OCB : OK cashbag(SK플래닛이 제공하는 OK캐쉬백 결제)
- ONESTORECASH : 원스토어 캐쉬(원스토어 캐쉬 결제)
- COUPON : 원스토어 쿠폰(원스토어 쿠폰 결제)
- POINT : 원스토어 포인트(원스토어 포인트 결제)
- TELCOMEMBERSHIP : 통신사멤버십(통신사에서 제공하는 멤버십 결제)
- EWALLET : e-Wallet(e-Wallet 결제)
- BANKACCT : 계좌결제(일반 계좌결제)
- PAYPAL : 페이팔(페이팔이 제공하는 결제)
- MYCARD : 마이카드(소프트월드에서 제공하는 마이카드 결제)

##### Signature 검증 방법
* 악의적인 의도를 가진 사용자가 임의의 메시지를 생성하여 전송할 수 있기 때문에 메세제에 대한 유효성 검증은 필수적입니다. 
아래 코드를 사용하면 signature에 대한 위변조 여부를 확인할 수 있습니다.

- 코드 내 PublicKey는 개발자센터의 '라이선스 관리' 메뉴에서 제공되는 라이선스 키를 의미합니다.

```Java
import java.security.PublicKey;
   
import org.apache.commons.codec.binary.Base64;
import org.codehaus.jackson.JsonNode;
import org.codehaus.jackson.map.ObjectMapper;
import org.codehaus.jackson.node.ObjectNode;
   
   
public class SignatureVerifier {
   
    private static final String SIGN_ALGORITHM = "SHA512withRSA";
    private ObjectMapper mapper = new ObjectMapper();
   
   
    boolean verify(String rawMsg, PublicKey key) throws Exception {
        // JSON 메시지에서 signature를 추출한다.
        JsonNode root = mapper.readTree(rawMsg);
        String signature = root.get("signature").getValueAsText();
        ((ObjectNode)root).remove("signature");
          
        // 추출한 signature가 올바른 값인지 검증한다.
        Signature sign = Signature.getInstance(SIGN_ALGORITHM);
        sign.initVerify(key);
        sign.update(root.toString().getBytes("UTF-8"));
        return sign.verify(Base64.decodeBase64(signature));
    }
}
```

```php
<?php
function formatPublicKey($publicKey) {
 $BEGIN= "-----BEGIN PUBLIC KEY-----";
 $END = "-----END PUBLIC KEY-----";
  
 $pem = $BEGIN . "\n";
 $pem .= chunk_split($publicKey, 64, "\n");
 $pem .= $END . "\n";
  
 return $pem;
}
  
function formatSignature($signature) {
 return base64_decode(chunk_split($signature, 64, "\n"));
}
  
// Sample message
$sampleMessage = '{"msgVersion":"3.1.0D","purchaseId":"SANDBOX3000000004564","developerPayload":"OS_000211234","clientId":"0000000001","productId":"0900001234","messageType":"SINGLE_PAYMENT_TRANSACTION","purchaseMillis":24431212233,"purchaseState":"COMPLETED","price":20000,"productName":"한글은?GOLD100(+20)","paymentTypeList":[{"paymentMethod":"DCB","amount":3000},{"paymentMethod":"ONESTORECASH","amount":7000}],"billingKey":"36FED4C6E4AC9E29ADAF356057DB98B5CB92126B1D52E8757701E3A261AF49CCFBFC49F5FEF6E277A7A10E9076B523D839E9D84CE9225498155C5065529E22F5","isTestMdn":true,"signature":"MNxIl32ws+yYWpUr7om+jail4UQxBUXdNX5yw5PJKlqW2lurfvhiqF0p4XWa+fmyV6+Ot63w763Gnx2+7Zp2Wgl73TWru5kksBjqVJ3XqyjUHDDaF80aq0KvoQdLAHfKze34cJXKR/Qu8dPHK65PDH/Vu6MvPVRB8TvCJpkQrqg="}';
  
// Sample public key
$publicKey = "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDMzpWJoK1GSOrr4juma5+sREYjdCW8/xSd9+6z6PAkUH5af97wy8ecfkLtP9LK5VskryfDlcOjfu0BgmHYntAqKT7B4KWk8jWbJ8VHUpp30H95UbcnCRFDqpEtwYzNA5gNMYKtAdbL41K8Fbum0Xqxo65pPEI4UC3MAG96O7X1WQIDAQAB";
  
  
// Parse JSON message
$jsonArr = json_decode($sampleMessage, true);
  
// Extract and remove signature
$signature = $jsonArr["signature"];
unset($jsonArr["signature"]);
$originalMessage = json_encode($jsonArr, JSON_UNESCAPED_UNICODE);
  
// Veify
$formattedKey = formatPublicKey($publicKey);
$formattedSign = formatSignature($signature);
$hash_algorithm = 'sha512';
  
$success = openssl_verify($originalMessage, $formattedSign, $formattedKey, $hash_algorithm);
if ($success == 1) {
 echo "verified";
}
else {
 echo "unverified";
}
?>
```

```Python
# -*- coding: utf-8 -*-
  
import json
from base64 import b64decode
from collections import OrderedDict
  
from Crypto.Hash import SHA512
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
  
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
  
hash = "SHA-512"
  
  
def verify(message, signature, pub_key):
    signer = PKCS1_v1_5.new(pub_key)
    digest = SHA512.new()
    digest.update(message)
    return signer.verify(digest, signature)
  
  
jsonData = json.loads(rawMsg, encoding='utf-8', object_pairs_hook=OrderedDict)
signature = jsonData['signature']
del jsonData['signature']
originalMessage = json.dumps(jsonData, ensure_ascii=False, encoding='utf-8', separators=(',', ':'))
  
RSA.importKey(publickey).publickey()
print(verify(originalMessage, b64decode(signature), RSA.importKey(publickey).publickey()))
```

##### Subscription Notification 메시지 발송 규격 (원스토어 → 개발사 서버) 
- URI : 개발자 센터에서 설정한 Subscription Notification URL
- Method : POST
- Request Parameters : N/A
- Request Header : 
```
Content-Type: application/json
```
- Request Body : JSON 형식
  - msgVersion : 메시지 버전(개발(Sandbox) : 3.1.0D, 상용(상용테스트) : 3.1.0)
  - clientId : 앱의 클라이언트 ID
  - eventTimeMillis : 이벤트 발생 시간
  - subscriptionNotification : 결제 정보 목록
    - version : 구독 알람 메시지 버전
    - notificationType : 구독상태
    - purchaseToken : 구매 토큰
    - productId : 인앱상품의 상품 ID
  - environment : 결제환경(개발(샌드박스) : SANDBOX, 상용 :COMMERCIAL)
  - marketCode : 마켓 구분코드 ( MKT_ONE : 원스토어, MKT_GLB : 글로벌 원스토어 

- Example

```json
{
    "msgVersion":"3.1.0",
    "clientId":"0000000001",
    "eventTimeMillis":24431212233000,
    "subscriptionNotification": {
        "version": "1",
        "notificationType" : 1,
        "purchaseToken":"TOKEN",
        "productId": "com.product.id"
    },
    "environmenmt": "COMMERCIAL",
    "marketCode": "MKT_ONE"
}
```

##### 구독상태 정의
[
  {"subscriptionStatus": "1", "subscriptionCode": "SUBSCRIPTION_RECOVERED", "description": "정기 결제가 보류 상태에서 복구 되었습니다."},
  {"subscriptionStatus": "2", "subscriptionCode": "SUBSCRIPTION_RENEWED", "description": "정기 결제가 갱신 되었습니다."},
  {"subscriptionStatus": "3", "subscriptionCode": "SUBSCRIPTION_CANCELED", "description": "고객이 정기 결제 해지를 요청 하였습니다."},
  {"subscriptionStatus": "4", "subscriptionCode": "SUBSCRIPTION_PURCHASED", "description": "새로운 정기 결제 상품이 구매 되었습니다."},
  {"subscriptionStatus": "5", "subscriptionCode": "SUBSCRIPTION_ON_HOLD", "description": "결제 실패로 정기 결제가 보류 상태가 되었습니다."},
  {"subscriptionStatus": "6", "subscriptionCode": "SUBSCRIPTION_IN_GRACE_PERIOD", "description": "결제 실패로 정기 결제가 유예 상태가 되었습니다."},
  {"subscriptionStatus": "7", "subscriptionCode": "SUBSCRIPTION_RESTARTED", "description": "고객이 정기 결제 해지 요청을 취소 하였습니다."},
  {"subscriptionStatus": "8", "subscriptionCode": "SUBSCRIPTION_PRICE_CHANGE_CONFIRMED", "description": "사용자가 정기 결제 가격 변경에 동의 하였습니다."},
  {"subscriptionStatus": "9", "subscriptionCode": "SUBSCRIPTION_DEFERRED", "description": "정기 결제 이용 기간이 연장 되었습니다."},
  {"subscriptionStatus": "10", "subscriptionCode": "SUBSCRIPTION_PAUSED", "description": "정기 결제가 일시 중지 되었습니다."},
  {"subscriptionStatus": "11", "subscriptionCode": "SUBSCRIPTION_PAUSE_SCHEDULE_CHANGED", "description": "정기 결제 일시 중지 일정이 변경 되었습니다."},
  {"subscriptionStatus": "12", "subscriptionCode": "SUBSCRIPTION_REVOKED", "description": "정기 결제가 즉시 해지 되었습니다."},
  {"subscriptionStatus": "13", "subscriptionCode": "SUBSCRIPTION_EXPIRED", "description": "정기 결제가 보류 상태에서 복구 되었습니다."},
]


#### PNS Notification 전송 정책
원스토어의 PNS 서버는 HTTP(S) 요청을 통하여 개발사 서버로 notification을 전송합니다.

이때 개발사 서버는 notification를 정상적으로 수신했다는 의미로 HTTP Status Code를 200으로 응답하여야 합니다.

만약 네트워크 지연으로 인한 유실 또는 개발사 서버의 비정상적인 상황으로 HTTP Status Code를 200으로 응답받지 못한 경우, PNS서버는 notification 전송이 실패했다고 판단하고 3일간 최대 30회의 재전송을 수행하게 됩니다.

notification의 재전송은 아래의 예시와 같이 일정한 delay를 가진 후 실행되며, 재시도 회수가 많아지면 delay가 점차 늘어나는 구조로 되어 있습니다.

##### PNS 전송 Example
1. 0회차(최초) 매세지는 지연 0초에 2020-05-17 13:10:00에 전송
2. 1회차 메시지는 지연 30초에 2020-05-17 13:10:30에 전송
3. 2회차 메시지는 지연 120초에 2020-05-17 13:12:30에 전송
4. 3회차 메시지는 지연 270초에 2020-05-17 13:17:00에 전송
5. 4회차 메시지는 지연 480초에 2020-05-17 13:25:00에 전송
6. 이후 n회차 메시지를 개발사의 수신이 완료되는 시점까지 계속적으로 전송

### 08. 정기 결제 적용하기
#### 개요
정기 결제는 정해진 기간동안 사용자가 서비스를 이용할 수 있으며, 결제 주기에 따라 정기적으로 결제가 갱신됩니다. 

일반적으로 음악이나 영화, 혹은 게임들을 이용할 때 이와 같이 정기 결제 서비스를 사용하고 있습니다.   

개발자는 다양한 서비스를 제공하기 위해 하나의 앱 안에서 여러가지 정기 결제 상품을 구성할 수 있고, 사용자는 여러가지 정기 결제를 이용할 수 있습니다.  

또한, 신규 사용자를 확보하기 위해 할인 가격으로 정기 결제를 제공하거나 체험을 위한 무료 기간도 제공할 수 있습니다.  

마찬가지로 기존 사용자가 이용중인 정기 결제 상품을 변경 할 수 있도록 상품을 전환 시킬 수도 있습니다. 

#### 정기 결제 처리 하기 
사용자가 정기 결제 상품을 구매하면, 상품을 이용하는 동안 다양한 상태 변경을 거칠 수 있습니다. 

앱은 정기 결제의 상태를 확인하여, 각 상태 변경에 대응해야 합니다. 

정기 결제의 상태는 원스토어 인앱 결제 라이브러리의 PurchaseClient.queryPurchasesAsync() 또는 서버 API getSubscriptionDetail을 사용하여 확인 할 수 있습니다. 

PurchaseClient.queryPurchasesAsync()
getSubscriptionDetail
[
  {"상태": "구독중", "queryPurchasesAsync반환여부": "Y", "recurringState": 0, "getSubscriptionDetail반환여부": "Y", "expiryTimeMillis": "향후 일정", "autoRenewing": true},
  {"상태": "해지", "queryPurchasesAsync반환여부": "Y", "recurringState": 1, "getSubscriptionDetail반환여부": "Y", "expiryTimeMillis": "향후 일정", "autoRenewing": false},
  {"상태": "유예", "queryPurchasesAsync반환여부": "Y", "recurringState": 0, "getSubscriptionDetail반환여부": "Y", "expiryTimeMillis": "향후 일정(유예 기간 종료)", "autoRenewing": true},
  {"상태": "보류", "queryPurchasesAsync반환여부": "N", "recurringState": 0, "getSubscriptionDetail반환여부": "Y", "expiryTimeMillis": "종료됨(예상 만료 시간 종료 또는 유예 기간 종료(있는 경우))", "autoRenewing": true},
  {"상태": "일시중지", "queryPurchasesAsync반환여부": "N", "recurringState": 0, "getSubscriptionDetail반환여부": "Y", "expiryTimeMillis": "종료됨", "autoRenewing": true},
  {"상태": "만료", "queryPurchasesAsync반환여부": "N", "recurringState": 1, "getSubscriptionDetail반환여부": "Y", "expiryTimeMillis": "종료됨", "autoRenewing": false}
]

정기 결제 상태가 변경되면, SubscriptionNotification이 전송됩니다. 관련 내용은 PNS(Payment Notification Service) 이용하기를 참고하세요.

#### 정기 결제 구매 
사용자가 정기 결제를 구매하면 PurchaseClient.queryPurchasesAsync()에서 정기 결제 결과를 반환하고, SUBSCRIPTION_PURCHASED인 SubscriptionNotification이 전송됩니다. 

이 알림을 받으면 원스토어 인앱결제 API를 쿼리하여 정기 결제 상태를 업데이트 해야 합니다. 정기 결제 리소스는 다음 예시를 참고하세요. 

```json
{
+    "acknowledgementState": 0, // 새로운 정기결제를 구매한 경우에는 acknowledge가 필요합니다.
+    "autoRenewing": true,
    "paymentState": 1,
    "lastPurchaseId": "22071114040010115614",
    "linkedPurchaseToken": null,
    "priceAmount": "610",
    "priceAmountMicros": 610000000,
    "nextPriceAmount": "610",
    "nextPriceAmountMicros": 610000000,
    "nextPaymentTimeMillis": 1658106000000,
    "pauseStartTimeMillis": null,
    "pauseEndTimeMillis": null,
    "priceCurrencyCode": "KRW",
    "countryCode": "KR",
    "startTimeMillis": 1657515841000,
+    "expiryTimeMillis": 1658156399000,
    "autoResumeTimeMillis": null,
    "cancelledTimeMillis": null,
    "cancelReason": null,
    "promotionPrice": null,
    "priceChange": null
}
```

#### 정기 결제 갱신
정기 결제가 갱신되면 SUBSCRIPTION_RENEWED 알림이 전송됩니다. 

이 알림을 받으면 원스토어 인앱결제 API를 쿼리하여 새로운 만료 일시로 정기 결제 상태를 업데이트 해야 합니다.  정기 결제 리소스는 다음 예시를 참고하세요. 

```json
{
+  "acknowledgementState": 1,
+  "autoRenewing": true,
    "paymentState": 1,
    "lastPurchaseId": "22071411443210116308",
    "linkedPurchaseToken": null,
    "priceAmount": "610",
    "priceAmountMicros": 610000000,
    "nextPriceAmount": "610",
    "nextPriceAmountMicros": 610000000,
    "nextPaymentTimeMillis": 1658365200000,
    "pauseStartTimeMillis": null,
    "pauseEndTimeMillis": null,
    "priceCurrencyCode": "KRW",
    "countryCode": "KR",
    "startTimeMillis": 1657766672000,
+  "expiryTimeMillis": 1658501999000,
    "autoResumeTimeMillis": null,
    "cancelledTimeMillis": null,
    "cancelReason": null,
    "promotionPrice": null,
    "priceChange": null
}
```

월간 상품의 경우 익월 동일일에 결제가 됩니다. (3개월, 6개월 상품도 동일하게 동작합니다.) 

예를들어, 1월 15일에 정기 결제가 시작되었다면, 2월 15일에 다음 결제가 발생합니다. 

하지만 동일일이 없는 경우에는 그 달의 마지막 날이 결제 갱신일이 됩니다.  

예를들어, 1월 31일에 정기 결제가 시작되었다면, 2월 28일(혹은 29일)에 다음 결제가 발생하고, 3월 28일(혹은 29)에 그 다음 결제가 발생합니다. 

#### 정기 결제 만료
정기 결제가 만료되면 PurchaseClient.queryPurchasesAsync()에서 더 이상 반환되지 않으며 사용자는 정기 결제를 이용할 수 없게 됩니다. 

SUBSCRIPTION_EXPIRED 인 SubscriptionNotification도 정기 결제가 만료될 때 전송됩니다. 

이 알림을 받으면 원스토어 인앱결제 API를 쿼리하여 정기 결제 상태를 업데이트 해야 합니다. 정기 결제 리소스는 다음 예시를 참고하세요. 

```json
{
    "acknowledgementState": 1,
+    "autoRenewing": false,
    "paymentState": 1,
    "lastPurchaseId": "22071213191410115875",
    "linkedPurchaseToken": null,
    "priceAmount": "610",
    "priceAmountMicros": 610000000,
    "nextPriceAmount": "610",
    "nextPriceAmountMicros": 610000000,
    "nextPaymentTimeMillis": 1658192400000,
    "pauseStartTimeMillis": null,
    "pauseEndTimeMillis": null,
    "priceCurrencyCode": "KRW",
    "countryCode": "KR",
    "startTimeMillis": 1657599554000,
+    "expiryTimeMillis": 1658242799000,
    "autoResumeTimeMillis": null,
    "cancelledTimeMillis": null,
    "cancelReason": null,
    "promotionPrice": null,
    "priceChange": null
}
```

#### 정기 결제 해지 
정기 결제는 사용자가 원스토어 앱에서 정기 결제를 직접 해지할 수 있고, 결제 수단에 문제가 생겼을 때 이를 해결하지 않는 경우 자동으로 해지 될 수 있습니다.    

원스토어 인앱결제 API cancelRecurringPruchase 를 사용하여 정기 결제를 해지할 수도 있습니다. 

사용자가 정기 결제를 직접 해지 하더라도 현재 정기 결제 기간이 끝날 때까지는 서비스를 이용할 수 있어야 합니다.  

정기 결제가 해지 되었지만 아직 만료되지 않은 경우 PurchaseClient.queryPurchasesAsync() 에서 반환됩니다. 정기 결제가 해지되면 SUBSCRIPTION_CANCELED 알림이 전송됩니다. 

이 알림을 받으면 원스토어 인앱결제 API를 쿼리하여 상태를 업데이트 해야 합니다. 쿼리 시 autoRenewing=false와 사용자가 서비스를 이용 할 수 있는 기한 expiryTimeMillis 이 반환 됩니다. 

expiryTimeMillis가 기한이 지난 과거라면 즉시 서비스가 중단되고, 기한이 미래라면 해당일시까지 서비스를 이용 할 수 있습니다. 정기 결제 리소스는 다음 예시를 참고하세요.     

```json
{
    "acknowledgementState": 1,
+    "autoRenewing": false,
    "paymentState": 1,
    "lastPurchaseId": "22071118111610115712",
    "linkedPurchaseToken": null,
    "priceAmount": "610",
    "priceAmountMicros": 610000000,
    "nextPriceAmount": "610",
    "nextPriceAmountMicros": 610000000,
    "nextPaymentTimeMillis": 1658106000000,
    "pauseStartTimeMillis": null,
    "pauseEndTimeMillis": null,
    "priceCurrencyCode": "KRW",
    "countryCode": "KR",
    "startTimeMillis": 1657515841000,
+    "expiryTimeMillis": 1658156399000,
    "autoResumeTimeMillis": null,
+    "cancelledTimeMillis": 1658156399000,
+    "cancelReason": 1,
    "promotionPrice": null,
    "priceChange": null
}
```
사용자에게 정기 결제가 해지 되었으며, 해당 정기 결제의 만료일시를 앱을 통해 알려야합니다. (예시 : 정기결제가 해지되었습니다. 2023년 6월 13일에 정기 결제가 만료됩니다. [확인])  

#### 정기 결제 취소
다양한 이유로 사용자가 정기 결제를 취소 할 수 있습니다. 정기 결제가 취소되면 즉시 서비스 이용이 차단되어야 합니다. 

이 경우 PurchaseClient.queryPurchasesAsync()에서 더이상 반환되지 않으며, SUBSCRIPTION_REVOKED 알림도 전송 됩니다. 

이 알림을 받으면 원스토어 인앱결제 API 를 쿼리하여 정기 결제 상태를 업데이트 해야 합니다. 

정기 결제 리소스에는 autoRenewing와 정기 결제 이용 기한인 expiryTimeMillis가 포함됩니다. 정기 결제 리소스는 다음 예시를 참고하세요. 

```json
{
    "acknowledgementState": 1,
+    "autoRenewing": false,
    "paymentState": null,
    "lastPurchaseId": "22071216245010115987",
    "linkedPurchaseToken": null,
    "priceAmount": "610",
    "priceAmountMicros": 610000000,
    "nextPriceAmount": "610",
    "nextPriceAmountMicros": 610000000,
    "nextPaymentTimeMillis": 1658192400000,
    "pauseStartTimeMillis": null,
    "pauseEndTimeMillis": null,
    "priceCurrencyCode": "KRW",
    "countryCode": "KR",
    "startTimeMillis": 1657610690000,
+    "expiryTimeMillis": 1657610749000,
    "autoResumeTimeMillis": null,
+    "cancelledTimeMillis": 1657610749000,
+    "cancelReason": 1,
    "promotionPrice": null,
    "priceChange": null
}
```

#### 정기 결제 유예 
사용자가 정기 결제 이용 중 결제 수단에 문제가 발생하고 문제가 해결되지 않는 경우 정기 결제가 바로 해지 되지는 않습니다. 

위와 같은 경우 (1) 결제 유예 - 유예 기간이 설정된 경우  (2) 계정 보류 순서대로 정기 결제 상태가 변경됩니다.  

결제 유예 기간 제공여부는 원스토어 개발자센터에서 설정 할 수 있습니다.  

유예 기간을 설정 했다면 사용자는 해당 기간 동안 정기 결제 콘텐츠를 이용할 수 있어야 합니다. 

앱에서 PurchaseClient.queryPurchasesAsync() 를 사용하여 정기 결제 상태를 확인한다면, PurchaseClient.queryPurchasesAsync()가 만료일 전에 정기 결제 상태를 계속 반환하므로 앱은 유예 기간을 자동으로 처리해야 합니다.  

사용자의 정기 결제 상태가 결제 유예가 되는 경우 SUBSCRIPTION_IN_GRACE_PERIOD 인 알림이 전송됩니다.

이 알림을 받으면 원스토어 인앱결제 API를 쿼리하여 상태를 업데이트 해야합니다. 정기 결제 리소스에는 autoRenewing=true 와 유예가 적용되도록 미래 시점의 expiryTimeMillis 이 포함됩니다. 

유예 기간 내에 결제 문제를 해결하지 않으면 계정 보류 상태가 되어 사용자가 정기 결제 콘텐츠를 이용 할 수 없습니다. 

앱에서는 메시지를 표시하여 사용자가 결제 문제를 해결 할 수 있도록 해야 합니다.  

사용자가 결제 문제를 해결하는데 도움이 되도록 딥링크를 제공할 수 있습니다.  

사용자가 결제 문제를 해결하면 즉시 정기 결제가 갱신 됩니다. 

```json
{
    "acknowledgementState": 0,
+    "autoRenewing": true,
    "paymentState": 0,
    "lastPurchaseId": "22071209533410115807",
    "linkedPurchaseToken": null,
    "priceAmount": "610",
    "priceAmountMicros": 610000000,
    "nextPriceAmount": "610",
    "nextPriceAmountMicros": 610000000,
    "nextPaymentTimeMillis": 1658192400000,
    "pauseStartTimeMillis": null,
    "pauseEndTimeMillis": null,
    "priceCurrencyCode": "KRW",
    "countryCode": "KR",
    "startTimeMillis": 1657587215000,
+    "expiryTimeMillis": 1658242799000,
    "autoResumeTimeMillis": null,
    "cancelledTimeMillis": null,
    "cancelReason": null,
    "promotionPrice": null,
    "priceChange": null
}
```

#### 정기 결제 계정 보류
사용자가 정기 결제 이용 중 결제 수단에 문제가 발생하고 문제가 해결 되지 않는 경우 유예 기간이 종료되면 계정 보류 상태가 됩니다. 

계정 보류 기간은 최대 30일이며, 유예 기간과는 다르게 계정 보류 상태에서는 정기 결제 콘텐츠를 이용 할 수 없습니다.

또한 계정 보류 중에는 정기 결제가 PurchaseClient.queryPurchasesAsync()에 의해 반환되지 않습니다. 

사용자의 정기 결제가 계정 보류 상태가 되면  SUBSCRIPTION_ON_HOLD 알림이 전송됩니다. 

이 알림을 받으면 원스토어 인앱결제 API를 호출하여 정기 결제 정보를 업데이트 해야합니다. 계정 보류 상태에서는 정기 결제 리소스의 expiryTimeMillis가 과거로 설정 됩니다. 

보류 상태에서는 사용자가 정기 결제 콘텐츠를 이용 할 수 없고, 지정된 보류 기간이 지나면 정기 결제가 해지됩니다. 

따라서 앱에서는 메시지를 표시하여 사용자가 문제를 해결 할 수 있도록 해야합니다. 

사용자가 문제를 해결하는데 도움이 되도록 딥링크를 제공할 수 있습니다.  

사용자가 문제를 해결하여 정기 결제가 복구되면 정기 결제일은 복구된 날짜로 변경됩니다.    

계정 보류 기간이 종료되기 전에 사용자가 문제를 해결하지 않으면 SUBSCRIPTION_CANCELED 알림이 전송 되며, 원스토어 인앱결제 API를 쿼리하여 정기 결제 상태를 업데이트 해야합니다. 

```json
{
    "acknowledgementState": 0,
+    "autoRenewing": true,
    "paymentState": 0,
    "lastPurchaseId": "22071209361310115799",
    "linkedPurchaseToken": null,
    "priceAmount": "610",
    "priceAmountMicros": 610000000,
    "nextPriceAmount": "610",
    "nextPriceAmountMicros": 610000000,
    "nextPaymentTimeMillis": 1659315600000,
    "pauseStartTimeMillis": null,
    "pauseEndTimeMillis": null,
    "priceCurrencyCode": "KRW",
    "countryCode": "KR",
    "startTimeMillis": 1657586174000,
+    "expiryTimeMillis": 1658242799000,
    "autoResumeTimeMillis": null,
    "cancelledTimeMillis": null,
    "cancelReason": null,
    "promotionPrice": null,
    "priceChange": null
}
```

#### 정기 결제 일시 중지
개발자는 원스토어 개발자센터에서 일시 중지 기능을 사용하도록 설정할 수 있습니다. 

일시 중지 기능을 사용하도록 설정하면, 사용자는 앱의 구독을 취소하는 대신 일정 기간동안 정기 결제를 일시 중지 할 수 있습니다. 

일시 중지는 현재 정기 결제 중인 상품의 기한이 만료된 후 시작됩니다. 일시중지 된 동안에는 사용자가 정기 결제 콘텐츠를 이용할 수 없어야 합니다. 

일시중지 기간이 끝나면 정기 결제가 다시 시작됩니다. 

사용자는 일시중지 기간이 끝나기 전에 직접 일시중지를 해제하여 정기 결제를 다시 시작할 수도 있습니다. 이 경우 일시중지를 해제하여 정기 결제를 다시 시작한 날짜가 정기 결제일이 됩니다. 

일시중지 중에는 PurchaseClient.queryPurchasesAsync()에 의해 반환되지 않습니다.  정기 결제가 재개되면 PurchaseClient.queryPurchasesAsync()에 의해 반환됩니다. 

사용자가 일시중지를 선택하면 SUBSCRIPTION_PAUSE_SCHEDULE_CHANGED인 SubscriptionNotification이 전송됩니다.  

이 시점에 사용자는 정기 결제 이용기간이 남아있으며, 따라서 정기 결제 콘텐츠를 계속 이용할 수 있어야 합니다.  

정기 결제 리소스에는 autoRenewing=true, paymentState=1과 미래의 expiryTimeMillis, autoResumeTimeMillis 값이 포함됩니다. 

```json
{
    "acknowledgementState": 1,
+    "autoRenewing": true,
+    "paymentState": 1,
    "lastPurchaseId": "22071114040010115614",
    "linkedPurchaseToken": null,
    "priceAmount": "610",
    "priceAmountMicros": 610000000,
    "nextPriceAmount": "610",
    "nextPriceAmountMicros": 610000000,
    "nextPaymentTimeMillis": 1660698000000,
+    "pauseStartTimeMillis": 1660748400000,
+    "pauseEndTimeMillis": 1663340399000,
    "priceCurrencyCode": "KRW",
    "countryCode": "KR",
    "startTimeMillis": 1657515841000,
+    "expiryTimeMillis": 1660748399000,
+    "autoResumeTimeMillis": 1660698000000,
    "cancelledTimeMillis": null,
    "cancelReason": null,
    "promotionPrice": null,
    "priceChange": null
}
```

일시중지가 시작되면 SUBSCRIPTION_PAUSED 인 Subscription Notification이 전송됩니다. 

일시중지가 시작되면 정기 결제 이용기간이 만료 된 것이며, 따라서 정기 결제 콘텐츠는 이용할 수 없어야 합니다. 

정기 결제 리소스에는 autoRenewing=true, paymentState=0과 과거의 expiryTimeMillis, autoResumeTimeMillis 값이 포함됩니다. 

일시중지 기간이 끝나서 정기 결제가 자동으로 재개되거나 사용자가 일시중지를 해제하여 정기 결제를 재개하면 SUBSCRIPTION_RENEWED 인 SubscriptionNotification이 전송됩니다. 

이 경우 정기 결제 갱신과 동일한 처리가 필요합니다. 

정기 결제가 재개될 때 결제에 문제가 생기면 정기 결제 계정 보류 상태가 발생하며, 정기 결제 계정 보류 상태와 동일하게 처리되어야 합니다. 

```json
{
    "acknowledgementState": 1,
+    "autoRenewing": true,
+    "paymentState": 0,
    "lastPurchaseId": "22071114040010115614",
    "linkedPurchaseToken": null,
    "priceAmount": "610",
    "priceAmountMicros": 610000000,
    "nextPriceAmount": "610",
    "nextPriceAmountMicros": 610000000,
    "nextPaymentTimeMillis": 1663290000000,
+    "pauseStartTimeMillis": 1660748400000,
+    "pauseEndTimeMillis": 1663340399000,
    "priceCurrencyCode": "KRW",
    "countryCode": "KR",
    "startTimeMillis": 1657515841000,
+    "expiryTimeMillis": 1660748399000,
+    "autoResumeTimeMillis": 1663290000000,
    "cancelledTimeMillis": null,
    "cancelReason": null,
    "promotionPrice": null,
    "priceChange": null
}
```

#### 상품 변경(업그레이드, 다운그레이드)
사용자가 정기 결제 중인 상품을 다른 정기 결제 상품으로 변경하면 기존에 사용 중인 정기 결제는 무효화되고, 새로운 정기 결제가 새 구매 토큰과 함께 생성됩니다. 

또한 정기 결제 리소스에는 기존 정기 결제를 나타내는 linkedPurchaseToken 이 포함됩니다.

linkedPurchaseToken 을 사용하면 이전 정기 결제를 조회하여 새로운 정기 결제를 동일한 계정과 연결할 수 있습니다. 

새로운 정기 결제가 생성 되었으므   로 해당 결제 건에 대한 구매 확인이 필요합니다. 정기 결제 리소스는 다음 예시를 참고하세요.  

```json
{
  "acknowledgementState": 1,
  "autoRenewing": true,
  "paymentState": 1,
  "lastPurchaseId": "22071214572510115940",
+  "linkedPurchaseToken": "220712131914S0115875",
  "priceAmount": "600",
  "priceAmountMicros": 600000000,
  "nextPriceAmount": "600",
  "nextPriceAmountMicros": 600000000,
  "nextPaymentTimeMillis": 1660266000000,
  "pauseStartTimeMillis": null,
  "pauseEndTimeMillis": null,
  "priceCurrencyCode": "KRW",
  "countryCode": "KR",
  "startTimeMillis": 1657605449000,
  "expiryTimeMillis": 1660316399000,
  "autoResumeTimeMillis": null,
  "cancelledTimeMillis": null,
  "cancelReason": null,
  "promotionPrice": null,
  "priceChange": null
}
```

#### 정기 결제 관리 화면 열기   
원스토어는 사용자가 정기 결제 상품을 쉽게 관리하도록 정기 결제 관리 메뉴를 제공합니다.  

매개변수로 SubscriptionsParams에 PurchaseData를 포함해서 넣으면 구매 데이터를 확인하여 해당 정기 결제 상품의 관리 화면을 실행합니다.
SubscriptionParams을 null 넣을 경우 사용자의 정기 결제 리스트 화면을 실행합니다.

다음은 정기 결제 관리 화면을 띄우는 방법을 나타내는 예제입니다.

```kotlin
fun launchManageSubscription(@Nullable purchaseData: PurchaseData) {
    val subscriptionParams = when (purchaseData != null) {
        true -> SubscriptionParams.newBuilder()
                    .setPurchaseData(purchaseData)
                    .build()
        else -> null
    }
    purchaseClient.launchManageSubscription(mActivity, subscriptionParams)
}
```
```java
public void launchManageSubscription(@Nullable PurchaseData purchaseData) {
    SubscriptionParams subscriptionParams = null;
    if (purchaseData != null) {
        subscriptionParams = SubscriptionParams.newBuilder()
            .setPurchaseData(purchaseData)
            .build();
    }
    purchaseClient.launchManageSubscription(mActivity, subscriptionParams);
}
```


- 규격 Uri : onestore://common/subscription/payment/{caller_package}?purchase_token={purchase_token}
- 규격 parameter : caller_package(mandatory) (연동 규격을 호출 할 앱의 packageName)
- 규격 parameter : purchase_token(optional) (특정 구매 상세로 가기위한 구매 Token)
- 상세 정보 : 마이페이지 > 정기결제 목록 화면으로 이동한다. purchase_token 이 존재하는 경우 해당 구매 Token의 정기결제 상세로 이동한다. 
- 지원버전 : ONE store Client v7.9.0 이상 ( android:versionName=”7.9.0” android:versionCode=”70900” ), ONE store Service v7.140 이상

#### 정기 결제 상품 변경 하기 
사용자는 정기 결제 이용 중에 좀 더 좋은 혹은 저렴한 정기 결제 상품으로 플랜을 변경하고 싶을 수 있습니다. 

사용자는 다른 정기 결제 상품을 결제하여 정기 결제 상품을 변경 할 수 있고, 개발자는 비례 배분 모드를 설정하여 정기 결제 상품 변경을 처리 할 수 있습니다. 

설정 할 수 있는 비례 배분 모드(PurchaseFlowParams.ProrationMode)는 다음과 같습니다. 

[
  {"비례배분모드": "IMMEDIATE_WITH_TIME_PRORATION", "설명": "정기 결제 상품이 즉시 변경 되며, 가격 차이를 기반으로 정기 결제 갱신일이 조정됩니다."},
  {"비례배분모드": "IMMEDIATE_AND_CHARGE_PRORATED_PRICE", "설명": "정기 결제 상품이 즉시 변경 되며, 정기 결제 갱신일은 동일하게 유지됩니다. 나머지 기간에 대한 가격 차이를 기반으로 요금이 청구됩니다.(이 모드는 상품 업그레이드 시에만 적용 가능합니다.) "},
  {"비례배분모드": "IMMEDIATE_WITHOUT_PRORATION", "설명": "정기 결제 상품이 즉시 변경 되며, 다음 결제일에 새로운 가격이 청구됩니다. 정기 결제 갱신일은 동일하게 유지됩니다."},
  {"비례배분모드": "DEFERRED", "설명": "기존 정기 결제 상품이 만료되면 교체가 적용되며 새로운 정기 결제 상품 요금이 청구됩니다."},
]


비례 배분 예

사용자는 현재 정기 결제 상품 A를 매월 정기 결제하고 있습니다. 이 정기 결제는 2,000원 의 요금이 청구되며 매월 1일에 갱신 됩니다.

4월 15일에 사용자는 요금이 연간 36,000원 인 연간 정기 결제 상품 B로 상품을 변경 하기로 했습니다.

#### IMMEDIATE_WITH_TIME_PRORATION
정기 결제 상품 A가 즉시 종료됩니다. 사용자는 한 달(4월 1일~30일) 요금을 결제 했는데 정기 결제 기간 중간에 상품을 변경 했으므로 월간 정기 결제 요금의 절반(1,000원)은 새 정기 결제에 적용됩니다. 그러나 새 정기 결제 요금은 연간 36,000원이므로 1,000원의 잔액은 10일(4월 16일~25일)에 해당합니다. 따라서 4월 26일에 새 정기 결제의 요금으로 36,000원이 청구되며, 다음 해부터 매년 4월 26일에 36,000원이 청구됩니다. 

#### IMMEDIATE_AND_CHARGE_PRORATED_PRICE
시간 단위 당 정기 결제 상품 B의 가격(36,000원/연 = 3,000원/월)이 정기 결제 상품 A의 가격(2,000원/월)보다 높으므로 이 모드를 사용할 수 있습니다.  정기 결제 상품A는 즉시 종료됩니다. 사용자는 한 달 전체 요금을 결제 했는데 기간의 절반만 사용했으므로 월간 정기 결제 요금의 절반(1,000원)은 새 정기 결제에 적용됩니다. 그러나 새 정기 결제 요금은 1년에 36,000원 이므로 나머지 15일의 요금은 1,500원입니다. 따라서 새 정기 결제 요금으로 500원의 차액이 청구됩니다. 5월 1일에는 새 정기 결제 상품 B의 요금으로 36,000원이 청구되며, 다음 해부터 매년 5월 1일에 36,000원이 청구됩니다. 

#### IMMEDIATE_WITHOUT_PRORATION
정기 결제 상품A가 추가 비용 없이 정기 결제 상품B로 즉시 변경 됩니다. 그리고 5월 1일에 새 정기 결제 상품 B의 요금으로 36,000원이 청구되며, 다음 해부터 매년 5월 1일에 36,000원이 청구됩니다.

#### DEFERRED
정기 결제 상품A는 4월 30일에 만료될 때까지 계속됩니다. 5월 1일에 정기 결제 상품 B가 적용되며, 새 정기 결제 등급 요금으로 사용자에게 36,000원이 청구됩니다.

정기 결제는 "구매 요청하기"와 동일한 API를 사용하여 사용자에게 상품 변경 기능을 제공할 수 있습니다. 다만, 정기 결제의 상품 변경을 위해선 기존 정기 결제 구매 토큰과 비례 배분 모드 값이 필수로 필요합니다.

다음 예와 같이 현재 정기 결제와 변경하여 적용 될 정기 결제 및 비례 배분 모드에 관한 정보를 제공해야 합니다.

```Kotlin
val subscriptionUpdateParams = SubscriptionUpdateParams.newBuilder()
      .setProrationMode(desiredProrationMode)
      .setOldPurchaseToken(oldPurchaseToken)
      .build()

val purchaseFlowParams = PurchaseFlowParams.newBuilder()
      .setProductId(newProductId)
      .setProductType(productType)
      .setProductName(productName)        // optional
      .setDeveloperPayload(devPayload)    // optional
      .setSubscriptionUpdateParams(subscriptionUdpateParams)
      .build()
```

```Java
SubscriptionUpdateParams subscriptionUpdateParams = SubscriptionUpdateParams.newBuilder()
      .setProrationMode(desiredProrationMode)
      .setOldPurchaseToken(oldPurchaseToken)
      .build();

PurchaseFlowParams purchaseFlowParams = PurchaseFlowParams.newBuilder()
      .setProductId(newProductId)
      .setProductType(productType)
      .setProductName(productName)        // optional 
      .setDeveloperPayload(devPayload)    // optional
      .setSubscriptionUpdateParams(subscriptionUdpateParams)
      .build();

purchaseClient.launchPurchaseFlow(activity, purchaseFlowParams);
```

purchaseClient.launchPurchaseFlow(activity, purchaseFlowParams)
즉시 변경이 이루어지는 비례 배분 모드의 경우 구매 요청하기 로직을 수행하기 때문에 응답은 PurchasesUpdatedListener에서 수신합니다. 또한 구매내역 조회하기에서도 요청 시 응답을 받을 수 있습니다.

 PurchaseClient.acknowledgeAsync()또는, acknowledgePurchase를 사용하여 구매 확인처리를 해야 합니다.

원스토어 인앱결제 API는 정기 결제 리소스에서 linkedPurchaseToken 을 반환합니다. linkedPurchaseToken 을 무효화하여 콘텐츠 이용 권한을 얻는 데 이전 토큰이 사용되지 않도록 해야 합니다. 

지연 변경이 이루어지는 비례 배분 모드의 경우 정기 결제 구매 및 변경 여부와 함께 PurchasesUpdatedListener  호출을 수신합니다. 

변경이 적용될 때까지 PurchaseClient.queryPurchasesAsync() 는 기존 정기 결제 상품 정보의 구매 정보를 계속 반환합니다.

변경이 적용되면 PurchaseClient.queryPurchasesAsync() 는 새로운 정기 결제 상품의 구매 정보를 반환하고, SUBSCRIPTION_RENEWED 인 Subscription Notification 이 전송됩니다.

지연 변경이 이루어지는 경우 이 알림을 수신하여 acknowledgePurchase 을 사용하여 구매확인 처리를 하는 것이 좋습니다. 

##### 프로모션이 있는 상품으로 변경하기 

원스토어에서는 프로모션이 적용되어 있는 정기 결제 상품을 이용 중인 경우 다른 상품으로 변경 할 수 없습니다. 

다만, 변경 대상이 되는 상품은 프로모션이 적용되어 있어도 변경이 가능합니다. 

변경 대상이 되는 상품의 프로모션을 고객이 계속 이용하려면 IMMEDIATE_WITH_TIME_PRORATION 모드를 적용해야 합니다. 

그 외 다른 비례 배분 모드의 경우 변경 대상이 되는 상품에 프로모션이 적용 되어 있더라도, 프로모션을 적용 받을 수 없습니다. 

#### 정기 결제 기한 연기하기  
원스토어 인앱결제 API의 deferSubscription 을 사용하여 정기 결제 사용자의 다음 결제일을 연기할 수 있습니다. 

연기 기간 동안 사용자는 콘텐츠에 엑세스 할 수 있습니다. 

정기 결제 기한 연기는 다음과 같은 경우 사용 할 수 있습니다. 

특별 이벤트를 통해 기존 사용자에게 일주일 동안 추가로 정기 결제를 사용할 수 있게 해줍니다.

시스템 장애 등의 사유로 고객에게 추가 이용 기간을 부여할 수 있습니다. 

API 호출 당 최소 하루에서 최대 1년까지 결제 기한을 연기 할 수 있습니다. 결제를 더 연기하려면 기한 내에 다시 API를 호출하면 됩니다. 

결제 연기 시 이메일 혹은 앱 내에서 알림을 보내 결제일이 변경 되었음을 알릴 수 있습니다. 

#### 정기 결제 가격 변경하기
원스토어 개발자센터에서 정기 결제 상품의 가격을 변경 할 수 있습니다.   

정기 결제 상품의 기본 가격을 변경하면, 새로운 구매는 즉시 변경된 가격으로 반영됩니다. 

다만, 현재 정기 결제를 사용중인 고객의 경우 7일 후에 변경된 가격에 대한 알림이 전달 되며, 이후 30일 간 가격 변경에 대한 동의 기간이 제공됩니다. 

> 단, Deferred 모드로 변경 예정인 구독 상품의 가격이 변경 된 경우에는
상품이 변경 된 후 가격 변경 동의 흐름이 시작 됩니다.

기간 내에 가격 변경에 대한 동의가 이루어지지 않으면, 해당 정기 결제는 다음 정기 결제일에 취소 됩니다. 

(가격이 내려가거나 유지되는 경우 별도의 알림이나 동의 절차 없이 7일 이후 다음 갱신일에 변경된 가격으로 결제 됩니다.)

##### 가격 변경에 대한 동의하는 경우  
사용자가 구독료 인상에 동의하는 경우 또는 가격을 인하한 경우에는 동의 기간 이후 변경된 가격으로 결제됩니다. 

또한 SUBSCRIPTION_PRICE_CHANGE_CONFIRMED 인 Subscription Notification이 전송됩니다. 

##### 가격 변경에 대해 동의하지 않는 경우 
사용자가 구독료 인상에 동의하지 않는 경우 자동으로 정기 결제가 취소되고, SUBSCRIPTION_EXPIRED 인 Subscription Notification이 전송됩니다. 

##### 실수로 가격 변경을 변경한 경우 
실수로 정기 결제 가격을 변경 했다면 가격을 다시 되돌리는 것이 좋습니다. 

7일 이내에 가격을 되돌린다면, 기존 정기 결제 사용자에게 영향을 미치지 않습니다.

다만, 이 경우에도 신규 구매 고객은 변경 된 가격으로 결제하게 되므로 가격 변경은 신중해야 합니다. 

##### 두 번의 가격 변경이 발생한 경우 
첫 번째 가격 변경 이후 두 번째 가격 변경이 7일 이내라면, 첫 번째 가격 변경은 무효화되고 두 번째 가격 변경만 적용이 됩니다. 

다만, 7일 이후라면 사용자는 두 번의 가격 변경에 대한 동의를 해야합니다. 

가격 변경에 대한 동의 절차는 사용자의 불편을 초래하며, 해당 정기 결제 상품을 유지하는 고객이 줄어들 가능성이 높아집니다. 

다시 한 번 강조하지만 가격 변경은 신중해야 합니다. 

### 09. 원스토어 인앱결제 릴리즈 노트
원스토어 인앱결제 라이브러리 21.02.00 업데이트
IAP SDK 21.02.00 부터 아래와 같이 onestore:dev_option의 android:value 값을 설정하면, SDK와 연동되는 스토어 앱을 지정 할 수 있습니다.

```
<manifest>
    <application>
        <meta-data android:name="onestore:dev_option" android:value="onestore_01" />
    </application>
</manifest>
```

android:value 값에 대응되는 대상 국가 및 지역은 아래와 같습니다.
- onestore_00 : 대한민국 (South Korea) (기본값)
- onestore_01 : 싱가포르, 타이완 (Singapore, Taiwan)
- onestore_02 : 미국 – Digital Turbine (United States)

StoreEnvironment API 기능 추가

StoreEnvironment.getStoreType() API는 SDK가 탑재된 애플리케이션이 원스토어를 통해 설치되었는지를 판단하는 기능을 제공합니다.

Store Type 정의

해당 API는 StoreType을 반환하며, 아래 네 가지 값 중 하나를 가집니다.

[
  {"StoreType": "StoreType.UNKNOWN", "value": "0", "description": "앱 설치 스토어 정보를 알 수 없음 (APK 직접 설치, 출처 불명 등)"},
  {"StoreType": "StoreType.ONESTORE", "value": "1", "description": "ONE Store에서 설치됨 (또는 개발자 옵션이 활성화된 경우)"},
  {"StoreType": "StoreType.VENDING", "value": "2", "description": "Google Play Store에서 설치됨"},
  {"StoreType": "StoreType.ETC", "value": "3", "description": "기타 스토어에서 설치됨"},
]

####  원스토어 인앱결제 라이브러리 SDK V21.01 배포
원스토어 인앱결제 SDK V21.01이 배포 되었습니다. 

주요 변경 사항은 다음과 같습니다. 

- 앱 무결성을 체크하는 보안 솔루션과 충돌이 있는 문제가 해결 되었습니다. 

- 국내 원스토어 앱이 설치되어 있을 때, 글로벌 원스토어 앱 다운로드 설치 동선에서 발생하던 로딩 문제가 해결 되었습니다. 

- 개발 단계에서 개발 환경을 변경 할 수 있는 설정을 제공합니다.

#### 원스토어 인앱결제 라이브러리 API V7(SDK V21) 출시
원스토어 인앱결제 라이브러리 API V7(SDK V21.00)가 출시 되었습니다.  주요한 변경 사항은 다음과 같습니다. 

새로운 상품은 원스토어 인앱결제 라이브러리 API V7(SDK V21)를 적용해야 합니다.

원스토어 인앱결제 API V5(SDK V17)이상을 적용한 앱은 API V7(SDK V21)로 마이그레이션이 가능합니다.

원스토어 인앱결제 API V4(SDK V16)은 이후 버전과 전혀 다른 구조로 설계 되었기 때문에, 원스토어 인앱결제 API V6(SDK V19)로 마이그레이션 할 수 없습니다.


##### 구독형 상품 
SDK V19 이하 버전에서 사용하던 월정액 상품을 대신하여 구독형 상품을 제공합니다. 

구독형 상품은 사용자가 원스토어 앱 내에서 정기 결제 상품을 직접 관리 할 수 있는 다양한 기능을 제공하고 

신규 유저 유입을 위한 프로모션, 정기 결제 상태 변경에 대한 알림 기능 등 개발자를 위한 다양한 기능도 제공됩니다.  



##### 로그인 프로세스 변경 
기존에 사용되든 PurchaseClient.launchLoginFlowAsync()는 지원되지 않을 예정입니다.

신규로 만들어진 GaaSignInClient를 이용하여 로그인이 지원됩니다. 클래스명이 바뀌었지만 사용방식은 동일합니다.




##### 배포 방식 변경
기존에는 guthub을 통한 배포였으나, V21부터는 maven을 통해 SDK가 배포 됩니다. 

이제부터는 SDK의 업데이트가 훨씬 간편 해졌습니다. 

#### 원스토어 인앱결제 라이브러리 API V6(SDK V19) 출시
원스토어 인앱결제 라이브러리 API V6(SDK V19.00)가 출시 되었습니다.  주요한 변경 사항은 다음과 같습니다. 

새로운 상품은 원스토어 인앱결제 라이브러리 API V6(SDK V19)를 적용해야 합니다.

원스토어 인앱결제 API V5(SDK V17)을 적용한 앱은 API V6(SDK V19)로 마이그레이션이 가능합니다.

원스토어 인앱결제 API V4(SDK V16)은 이후 버전과 전혀 다른 구조로 설계 되었기 때문에, 원스토어 인앱결제 API V6(SDK V19)로 마이그레이션 할 수 없습니다.


##### 원스토어 인앱결제 API 버전 삭제 
이제 앱의 매니페스트에서 인앱 API 버전을 추가할 필요가 없습니다.

인앱결제 SDK V19에 API Version meta-data가 추가됩니다.

##### 구매 확인 
원스토어 인앱결제 라이브러리 API V6(SDK V19) 이상을 사용하는 앱에서는 3일 이내에, 구매 확인을 해야합니다. 

3일 이내에 구매 확인이 되지 않으면, 아이템이 정상적으로 지급이 되지 않은 것으로 판단하고 구매를 취소 합니다. 

다음 메서드 중 하나를 사용하여 구매를 확인할 수 있습니다.

- 소모성 제품인 경우 PurchaseClient.consumeAsync()를 사용합니다.

- 소모성 제품이 아니라면 PurchaseClient.acknowledgeAsync()를 사용합니다. 

월정액 상품의 경우에는 최초의 결제에 대해서만 구매를 확인하면 됩니다.  

원스토어 인앱결제 API V6(SDK V19)를 적용한 경우, 반드시 3일 이내에 구매 확인을 해야합니다.

3일 이내에 구매 확인이 되지 않은 경우 구매가 취소되니 주의해야 합니다.



##### 마켓 구분 코드 얻기
IAP 라이브러리 V6부터 Server to Server API를 사용하기 위해서는 마켓 구분 코드가 필요합니다.

getStoreInfoAsync()를 통해서 마켓 구분 코드를 획득 할 수 있습니다. 

#### PurchaseClient API 변경 
[
  {"function": "결제모듈과의 연결", "V5_SDKv17": "connect", "V6_SDKv19": "startConnection"},
  {"function": "결제모듈과의 연결해제", "V5_SDKv17": "terminate", "V6_SDKv19": "endConnection"},
  {"function": "결지원여부 확인", "V5_SDKv17": "isBillingSupportedAsync", "V6_SDKv19": null},
  {"function": "인앱 상품 구매", "V5_SDKv17": "launchPurchaseFlowAsync", "V6_SDKv19": "launchPurchaseFlow"},
  {"function": "인앱 상품 소비", "V5_SDKv17": "consumeAsync", "V6_SDKv19": "consumeAsync"},
  {"function": "인앱 상품 구매 확인", "V5_SDKv17": null, "V6_SDKv19": "acknowledgeAsync"},
  {"function": "소비되지 않은 상품의 구매내역(월 자동결제 포함)", "V5_SDKv17": "queryPurchasesAsync", "V6_SDKv19": "queryPurchasesAsync"},
  {"function": "인앱 상품 상세정보", "V5_SDKv17": "queryProductsAsync", "V6_SDKv19": "queryProductDetailsAsync"},
  {"function": "월 자동결제 상태 변경", "V5_SDKv17": "manageRecurringProductAsync", "V6_SDKv19": "manageRecurringProductAsync"},
  {"function": "결제 모듈 업데이트 또는 설치", "V5_SDKv17": "launchUpdateOrInstallFlow", "V6_SDKv19": "launchUpdateOrInstallFlow"},
  {"function": "원스토어 로그인 호출", "V5_SDKv17": "launchLoginFlowAsync", "V6_SDKv19": "launchLoginFlowAsync"},
  {"function": "마켓 구분 코드 확인", "V5_SDKv17": null, "V6_SDKv19": "getStoreInfoAsync"},
]

#### Server API 변경 
아래는 기존 V5(SDK V17)대비 V6(SDK V19)dptj 변경, 삭제, 추가된 기능의 정리이다.

- 변경
[
  {"feature": "OAuth 토큰발급", "V5ApiURI": "/v2/oauth/token", "v6ApiURI": "/v6/oauth/token", "변경사항": "응답규격 중 status 필드 삭제,Http response code로 정상(200) 여부 확인"},
  {"feature": "구매상품 상세조회 By 인앱상품ID", "V5ApiURI": "/v2/purchase/details-by-productid
/{purchaseId}/{packageName}/{productId}", "v6ApiURI": "/v6/apps/{packageName}/purchases/inapp/products
/{productId}/{purchaseToken}", "변경사항": "응답 규격에 purchaseId, acknowledgeState 필드 추가"},
  {"feature": "월정액 상품 구매 상세조회", "V5ApiURI": "/v2/purchase/recurring-details/{purchaseId}/{packageName}", "v6ApiURI": "/v6/apps/{packageName}/purchases/auto/products
/{productId}/{purchaseToken}", "변경사항": "응답규격에 acknowledgeState, lastPurchaseId, lastPurchaseState 추가, 응답규격에 price, developerPayload, purchaseState 삭제"},
  {"feature": "자동결제 해지", "V5ApiURI": "/v2/purchase/manage-payment-status/{purchaseId}/{packageName}/cancel", "v6ApiURI": "/v6/apps/{packageName}/purchases/auto/products
/{productId}/{purchaseToken}/cancel", "변경사항": "성공처리 응답규격변경"},
  {"feature": "자동결제 해지취소", "V5ApiURI": "/v2/purchase/manage-payment-status/{purchaseId}/{packageName}/reactivate", "v6ApiURI": "/v6/apps/{packageName}/purchases/auto/products
/{productId}/{purchaseToken}/reactivate", "변경사항": "성공처리 응답규격변경"},
   {"feature": "구매취소 내역조회", "V5ApiURI": "/v2/purchase/voided-purchases/{packageName}", "v6ApiURI": "/v6/apps/{packageName}/voided-purchases", "변경사항": "API 사용성 향상을 위하여 구매취소내역 조회기준(startTime, endTime)을 구매일시에서 구매취소일시로 변경, 응답규격에 purchaseToken, marketCode 추가"},
  {"feature": "구매상품 소비", "V5ApiURI": "/v2/purchase/consume/{purchaseId}/{packageName}", "v6ApiURI": "/v6/apps/{packageName}/purchases/all/products/
{productId}/{purchaseToken}/consume", "변경사항": "성공처리 응답규격변경"}
]

- 삭제
[
  {"feature": "구매상품 상세조회", "V5ApiURI": "/v2/purchase/details/{purchaseId}/{packageName}", "v6ApiURI": null, "변경사항": "보안 강화를 위하여 구매상품 상세조회 API는 삭제, 구매상품 상세조회 By 인앱상품ID로 통합"},
  {"feature": "월정액 상품 마지막 구매 상세조회", "V5ApiURI": "/v2/purchase/last-recurring-details/{purchaseId}/{packageName}", "v6ApiURI": null, "변경사항": "월정액 상품 구매 상세조회 API 로 통합"}
]

- 추가
[
  {"feature": "구매 확인", "V5ApiURI": null, "v6ApiURI": "/v6/apps/{packageName}/purchases/all/products/
{productId}/{purchaseToken}/acknowledge", "변경사항": "구매 확인 API 신규 추가"}
]

#### PNS 메시지 규격 변경 
- 원화 외의 통화를 지원하기 위하여 결제금액(price)의 데이터 타입이 Number에서 String으로 변경하였습니다.

- 원화 외의 통화를 지원하기 위하여 결제금액의 통화코드(priceCurrencyCode)를 추가하였습니다.

- 원화 외의 통화를 지원하기 위하여 결제수단별금액(amount)의 데이터 타입이 Number에서 String으로 변경하였습니다.

- 응답 규격에 purchaseToken, environment, marketCode 필드가 추가되었습니다. 

상세한 규격은 PNS 메시지 상세 변경 내역에서 확인 할 수 있습니다.  

#### 원스토어 인앱결제 라이브러리 API V5(SDK V17) 
- API 버전은 개발사 애플리케이션의 'AndroidManifest.xml' 파일에 아래와 같이 명시되어야 합니다.
- 자세한 내용은 인앱결제 적용을 위한 사전준비 페이지의 'Android Manifest 파일 설정' 부분을 참고하시기 바랍니다.
- API V5 (SDK V17)에서 개선된 사항은 다음과 같습니다.
  - API
    - Consume 개념 도입 : 아이템 미지급 상황 발생 시 앱에서 복구할 수 있는 로직을 제공합니다.
    - Custom PID 지원 : 인앱상품 ID(PID)를 개발자가 직접 입력할 수 있어, 직관적인 PID로 생성하여 운영할 수 있습니다.
    - 보편적인 인앱결제 인터페이스 제공 : 타 마켓향으로 개발한 앱을 코딩상의 큰 개발공수 없이 원스토어향으로 런칭할 수 있습니다.
  - 결제 테스트 환경
    - 개발/상용 테스트 환경 분리 제공 : 개발사의 검증 환경과 동일 수준의 원스토어 결제환경을 제공합니다. 개발사의 개발환경에서는 원스토어 Sandbox와 연동하여 개발/테스트 할 수 있고, 개발사의 상용환경에서는 원스토어 상용환경에서 실결제를 해볼 수 있습니다. 원스토어 개발자센터에서 테스트 ID로 등록한 ID로 원스토어 로그인 후, 실결제한 경우, 개발자가 자유롭게 구매취소 가능합니다.
  - 결제창 UI
    - 전면/팝업 결제화면 선택 가능 : 전체 결제화면 또는 팝업 결제화면을 선택할 수 있습니다. 결제화면 설정은 원스토어 인앱결제 적용을 위한 사전준비 페이지의 'Android Manifest 파일 설정' 부분을 참고하시기 바랍니다.
  - 연동 방식
    - IAP SDK, AIDL 선택 가능 : 기존의 SDK 방식 뿐만 아니라, AIDL를 이용하여 직접 OSS와 연동할 수 있습니다. 단, 원스토어 서비스(ONE store service, OSS) 앱이 설치되어 있지 않은 단말에서는 인앱결제가 정상적으로 동작하지 않습니다. 원스토어 서비스 앱 설치에 대한 내용은 원스토어 인앱결제 적용을 위한 사전준비 페이지의 '원스토어 앱 설치' 부분을 참고하시기 바랍니다.
  - 개발자센터
    - 타 마켓 앱 상품 정보 크롤링 기능 개선 : 기 등록된 타 마켓 앱 상품을 원스토어에 간편하게 등록할 수 있습니다.
    - 타 마켓 인앱상품 정보 import 기능 제공
  - 인증센터
    - OAuth 발급/인증/관리 : 개발자의 서버와 원스토어 인앱결제 서버 연동 시 데이터 보안을 강화하였습니다.

### 10. Sample App Download
#### 원스토어 인앱결제 Sample App 다운로드
원스토어 인앱결제 API V7(SDK V21)를 사용하기 위해 필요한 샘플 앱을 깃허브(github)에서 다운로드 받을 수 있습니다. 
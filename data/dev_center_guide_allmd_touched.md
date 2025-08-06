출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21.md
# 원스토어 인앱결제 API V7(SDK V21) 연동 안내 및 다운로드

원스토어의 최신 인앱결제 API V7(SDK V21)이 출시되었습니다.

보다 강력하고 다양한 기능을 지원하는 최신 버전을 적용해보세요.

{% hint style="info" %}
API V4(SDK V16) 이하 버전과는 호환되지 않습니다. 인앱결제 API V4(SDK V16)에 대한 안내 및 다운로드는 [여기](old-version/v16)를 클릭해주세요.
{% endhint %}

{% hint style="info" %}
현재 판매중인 앱을 대한민국 외 국가/지역으로 배포하기 위해서는 아래 가이드를 참고해주세요

* [대한민국 외 국가 및 지역 배포를 위한 가이드](../glb)
{% endhint %}

If you are comfortable with English, please change the language to English from the upper left side in this page.

* [01. 원스토어 인앱결제 개요](v21/ov)
* [02. 원스토어 인앱결제 적용을 위한 사전준비](v21/pre)
* [03. 결제 테스트 및 보안](v21/test)
* [04. 원스토어 인앱결제 SDK를 사용해 구현하기](v21/sdk)
* [05. 원스토어 인앱결제 레퍼런스](v21/references)
* [06. 원스토어 인앱결제 서버 API (API V7)](v21/serverapi)
* [07. PNS(Payment Notification Service) 이용하기](v21/pns)
* [08. 정기 결제 적용하기](v21/subs)
* [09. 원스토어 인앱결제 릴리즈 노트](v21/releasenote)
* [10. Sample App Download](v21/sample)
* [11. V21로 원스토어 인앱결제 업그레이드 하기](v21/upgrade)
* [12. Unity에서 원스토어 인앱결제 (SDK V21) 사용하기](v21/unity)
* [13. Unity에서 IAP SDK v21로 업그레이드 하기](v21/unitymig)
* [14. Flutter에서 원스토어 인앱결제 사용하기](v21/flutter)
* [15. 웹 결제 규격 적용하기](v21/web-payment)

\


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/ov.md
# 01. 원스토어 인앱결제 개요

## **원스토어 인앱결제란?** <a href="#id-01." id="id-01."></a>

원스토어 인앱결제(In-App Purchase, IAP)는 안드로이드 앱 내에 구현된 상품을 원스토어의 인증 및 결제 시스템을 이용하여 사용자에게 판매, 청구하여 개발자에게 정산하는 서비스입니다.

인앱 상품 결제를 위해 원스토어 서비스(ONE store service, OSS) 앱과 연동이 필요하며, 원스토어 서비스 앱은 원스토어 결제 서버와 연동하여 인앱상품의 결제를 수행합니다.

원스토어 인앱결제를 적용하기 위해서는 '원스토어 IAP SDK(In-App Purchase Software Development Kit)'를 적용하면 됩니다.

## **인앱 상품의 유형** <a href="#id-01." id="id-01."></a>

원스토어 인앱결제 API V7(SDK V21)는 관리형 상품과 구독형 상품. 두 가지 유형의 인앱 상품을 제공 합니다.\
우선, 각 타입이 어떤 특성을 가지고 있는지 확인하신 후 본인이 제공하는 상품에 맞게 인앱 상품 목록을 구성해 보시기 바랍니다.

<table><thead><tr><th width="239">구분</th><th>내용</th></tr></thead><tbody><tr><td>관리형 상품</td><td>구매 후 소비(consume)하기 전까지 재구매가 되지 않는 상품입니다. 해당 상품을 다시 구매하기 위해서는 소비(consume) 처리를 해야 합니다.<br>해당 특성을 이용하여 소비성/영구성/기간제 형태의 인앱 상품을 제공 할 수 있습니다.</td></tr><tr><td>월정액 상품 (Deprecated)</td><td><p>한 번 가입하면 지정된 날짜에 매월 일정 금액이 자동결제 되는 상품으로, 자동결제 갱신(재결제)은 원스토어에서 처리합니다.</p><p>SDK V21 (API V7) 적용 이후 신규 월정액 상품은 만들 수 없습니다. (기존에 만들어진 월정액 상품은 계속 이용 하실 수 있습니다.) </p><p>구독형 상품을 이용하여 정기 결제 상품을 제공할 수 있습니다,  </p></td></tr><tr><td>구독형 상품</td><td><p>SDK V21 (API V7) 부터 제공되는 정기 결제 상품 타입입니다.</p><p>정해진 주기에 따라 원스토어에서 정기 결제를 처리하며, 신규 고객을 유치하기 위한 프로모션 기능 및 결제 수단 변경 등의 사용자 편의 기능을 제공합니다. </p><p>구독형 상품에 대한 자세한 내용은 <a href="subs">정기결제 적용하기</a> 페이지를 참고하시기 바랍니다.</p></td></tr></tbody></table>

## **결제 프로세스** <a href="#id-01." id="id-01."></a>

원스토어 인앱결제는 크게 다음의 네 가지 프로세스로 구성되어 있습니다.&#x20;

### 원스토어 로그인 하기 <a href="#id-01." id="id-01."></a>

#### **GaaSignInClient 초기화**

GaaSignInClient는 원스토어 로그인을 하기 위한 라이브러리입니다.

getClient() 통해 인스턴스를 생성합니다.

```kotlin

  val signInClient = GaaSignInClient.getClient(activity)
 
```

#### **포그라운드 로그인하기**

slientSignIn()과 달리 해당 함수는 UiThread 에서만 호출해야 합니다.

기본적으로 먼저 백그라운드 로그인을 시도하지만 실패에 대한 처리를 SDK에서 전담하여 처리합니다.\
이 후, 로그인 화면을 띄워 사용자에게 로그인을 유도합니다.

```kotlin

  signInClient.launchSignInFlow(activity) { signInResult ->
  
  }
 
```

#### **백그라운드 로그인하기**

slientSignIn() 통해 백그라운도 로그인을 호출합니다.

사용자가 이미 원스토어 계정에 로그인 되어있을 경우 이후 부터는 백그라운드에서 토큰로그인을 시도합니다. 성공 또는 실패에 대한 결과값으로 SignInResult 객체로 응답을 받습니다.

```kotlin

  signInClient.silentSignIn { signInResult ->
  
  }
 
```

### **인앱 상품 구매 요청하기**  <a href="#id-01." id="id-01."></a>

상품을 구매하려면 먼저 상품 상세정보(queryProductDetailAsync) API 호출하여 상품의 정보을 가지고 launchPurchaseFlow API를 통해 구매 화면을 호출합니다.

결제가 완료되면 PurchaseClient 객체를 초기화할 때 입력한 PurchasesUpdatedListener를 통해 구매정보를 전달받을 수 있습니다.

결과 코드값을 검증한 후에 성공으로 확인되면 인앱 상품에 해당하는 아이템을 지급하는 프로세스를 진행합니다.

아래 그림은 개발사 앱과 원스토어 사이의 인앱 상품을 구매하는 프로세스를 나타낸 것입니다.&#x20;

&#x20;

<figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2Fp9qDWlNUqSDbGD8tvxQH%2Fimage.png?alt=media&#x26;token=6d15b347-fa3d-4b63-aab2-138df9135dc3" alt=""><figcaption></figcaption></figure>

### **구매 확인하기**  <a href="#id-01." id="id-01."></a>

원스토어 IAP 라이브러리 V6 이상을 사용하는 경우 3일 이내에 구매 확인을 진행해야 합니다. 구매 확인이 되지 않으면, 고객에게 구매 금액이 환불됩니다.&#x20;

다음 두 가지 방법을 사용하면 구매를 확인할 수 있습니다.&#x20;

*   **소비하기 (consume)**&#x20;

    * 관리형 상품의 특징은 구매 후 소비(consume) 하기 전까지 재구매가 되지 않는 것입니다. 따라서, 관리형 상품 중 반복 구매가 가능한 소비성 상품을 제공할 경우에는 반드시 PurchaseClient.consumeAsync()를 사용하여 소비 처리를 해야 고객이 해당 상품을 다시 구매할 수 있습니다. &#x20;
    * 소비성 형태의 상품을 구현할 때는 구매 프로세스 완료 후 바로 구매한 상품을 소비시키고, 소비가 완료된 후에 아이템을 지급합니다. &#x20;
    * 사용자가 앱(게임)을 실행하는 시점 또는 앱(게임)에 로그인하는 시점에 위의 API들을 호출하여, 미지급된 상품이 있는지 사전에 체크하고 상품을 지급하는 프로세스를 추가하는 것을 권장합니다.  \


    <figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2FaRwb57SLYpt27clACavA%2Fimage.png?alt=media&#x26;token=3a66175d-f6f3-4845-82c1-23484c8a4869" alt=""><figcaption></figcaption></figure>
* **구매 확인하기 (acknowledge)**
  * 소비성 상품이 아닌 경우, PurchaseClient.acknowledgeAsync() 를 사용합니다. 소비 처리를 하지않고, 구매 인증만 하는 방법으로 영구성 상품을 구현할 수 있습니다.&#x20;
  * 영구성 형태의 상품을 구현할 때는, 소비가 아닌 구매 확인 (acknowledge)을 해야 합니다.&#x20;
  *   구독 상품의 경우에도 구매 확인이 필요하며, 최초 결제에 대해서만 구매 확인을 진행하면 됩니다.  \


      <figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2FtpneRxlaTI4QskC895DK%2Fimage.png?alt=media&#x26;token=366d6ae4-828f-465e-b48d-50d887508798" alt=""><figcaption></figcaption></figure>

네트워크 단절이나 앱의 갑작스러운 비정상 종료 등으로 인해, 고객은 결제를 완료하였으나 개발사에 결제정보 전달이 누락되어 인앱 상품이 지급되지 않는 경우가 발생할 수 있습니다.\
구매정보 가져오기(queryPurchases) 및 소비(consume) 기능을 활용하여 구매가 완료 되었으나 미지급된 내역이 있는지 확인 후 미지급된 상품이 지급 처리될 수 있도록 프로세스를 구현하는 것을 권장합니다.&#x20;

### 구독형 상품 관리 <a href="#id-01." id="id-01."></a>

구독형 상품(subscription)은 첫 구매 이후 상품 결제 주기에 따라 정기적으로 결제를 갱신합니다.

원스토어는 사용자가 정기 결제 상품을 관리 할 수 있도록 일시 중지, 구독 해지, 결제 수단 변경 등 다양한 기능이 있는 정기 결제 관리 화면을 제공합니다.  &#x20;

launchManageSubscription API를 호출하거나 딥링크를 사용하여 정기 결제 관리 메뉴를 제공 할 수 있습니다.&#x20;

## **권장 개발 환경** <a href="#id-01." id="id-01."></a>

원스토어 IAP SDK를 적용하기 위해 다음과 같은 개발 환경이 필요합니다.

* Android 6.0 이상 버전(API 버전 23 이상)
* [Java SDK 11 버전](http://www.oracle.com/technetwork/java/javase/downloads/index.html)
* [Android studio 2.0 이상 버전](https://developer.android.com/studio/index.html)



출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/pre.md
# 02. 사전준비

## **상품 등록하기**  <a href="#id-02." id="id-02."></a>

* 상품 등록과 관련된 내용은 아래 가이드를 참고 부탁 드립니다.&#x20;
  * [상품 등록과 관리](../../../docs/apps)&#x20;

## **라이선스 키(Public Key) 및 OAuth 인증 정보 확인하기**  <a href="#id-02.-publickey-oauth" id="id-02.-publickey-oauth"></a>

공통정보 >  라이선스 관리 메뉴에서 라이선스 키와 서버 API를 위한 OAuth 인증 정보를 확인할 수 있습니다.&#x20;

* 라이선스 키 : 원스토어가 전달한 인앱결제 내역의 위변조 여부를 확인하는 용도로 사용합니다.
* OAuth 인증 정보 : 원스토어 서버 API를 사용하기 위한 인증 용도로 사용합니다.



## **원스토어 인앱결제 Sample App 다운로드** <a href="#id-02.-sampleapp" id="id-02.-sampleapp"></a>

원스토어 인앱결제를 사용하기 위해 필요한 샘플 앱을 [깃허브(github](https://github.com/ONE-store/onestore_iap_release))에서 다운로드 받을 수 있습니다.&#x20;

## **인앱결제 라이브러리 추가하기**  <a href="#id-02." id="id-02."></a>

프로젝트 최상위 _build.gradle_ 파일에 원스토어 maven 주소를 등록합니다.

```gradle

allprojects {
    repositories {
        ...
        maven { url 'https://repo.onestore.net/repository/onestore-sdk-public' }
    }
}

```

다음은 앱의 _build.gradle_ 파일에 원스토어 결제 라이브러리 종속 항목을 추가합니다.

```gradle

dependencies {
    implementation "com.onestorecorp.sdk:sdk-iap:21.02.01"
}

```



## **원스토어 앱 설치하기**  <a href="#id-02." id="id-02."></a>

개발자나 사용자가 원스토어 인앱결제를 이용하기 위해서는 원스토어 앱이 필요합니다.

* [원스토어 앱 다운로드 안내](../../../policy/download)



출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/test.md
# 03. 결제 테스트 및 보안

## **개요** <a href="#id-03." id="id-03."></a>

원스토어는 개발환경(이하 Sandbox) 또는 상용 결제환경(이하 상용테스트)에서 인앱상품의 결제 테스트를 지원합니다.

Sandbox는 상용 결제환경이 아닌 가상의 결제환경으로, 결제화면에서 결제실패 또는 성공 중 원하는 응답을 선택하면 해당 응답결과를 전송합니다.

상용테스트는 상용 결제환경에서 해당 인앱상품의 결제를 진행하여 결제 결과를 전송합니다.

상용테스트 환경에서 결제를 하실 경우 취소하지 않으면 과금이 발생할 수 있으니, 결제 테스트 후 반드시 결제취소를 진행해주셔야 합니다.

{% hint style="warning" %}
주의사항

* 상품을 배포하고자 하는 OS에서 반드시 한 번은 샌드박스 테스트를 진행해야 합니다.
* OGL AAB/APK의 경우 안드로이드 환경에서 샌드박스 테스트를 진행하면 됩니다.&#x20;
*   Sandbox 및 상용환경에서 테스트하기 위해서는 사전에 원스토어 ID를 테스트 ID로 등록해야 합니다.

    등록하지 않은 원스토어 ID로 테스트할 경우 Sandbox에서 테스트가 불가능하고, 상용테스트 환경에서는 테스트 상용결제가 아닌 실제 상용결제가 진행되어 과금이 되므로 주의해주시기 바랍니다.
* 원스토어는 테스트 ID를 통해 결제한 내역에 대해 책임지지 않습니다. 테스트 ID의 관리 및 결제 테스트는 개발사 내부 책임자의 관리감독 하에 진행하시기 바랍니다.
{% endhint %}

## **인앱결제 테스트 화면** <a href="#id-03." id="id-03."></a>

In-App정보 화면에서 '결제 테스트' 버튼을 누르면 인앱결제 테스트 화면이 제공됩니다.

인앱결제 테스트 화면에서 테스트 ID를 관리하거나, 결제 테스트 내역을 조회할 수 있습니다.

## **테스트 ID 등록/관리** <a href="#id-03.-id" id="id-03.-id"></a>

개발 및 상용환경에서 인앱결제를 테스트하기 위해서는 사전에 테스트 ID를 등록해야 합니다.

인앱결제 테스트 화면에서 '테스트 ID 등록/관리' 탭을 선택하시면, 테스트 ID를 관리하는 화면으로 이동합니다.

테스트 ID는 원스토어 사용자 ID로, 원스토어에 회원으로 등록되어 있는 ID를 조회하여 등록할 수 있습니다.&#x20;

만약, 원스토어 회원으로 등록되어 있지 않다면 원스토어 앱을 실행하여 테스트용 원스토어 ID를 만드신 후 해당 ID를 테스트 ID로 등록하면 됩니다. &#x20;

### **테스트 ID 등록하기**   <a href="#id-03.-id" id="id-03.-id"></a>

* 테스트 ID로 사용 할 원스토어 회원 ID를 조회합니다.
* 조회한 원스토어 ID 확인 후, 해당 테스트 ID로 어느 결제환경(Sandbox, 상용테스트)에서 결제 테스트를 진행할지 선택합니다.
*   필요한 경우 참고 사항을 입력하고 등록 버튼을 눌러 테스트 ID를 등록 합니다.

    하나의 테스트 ID는 하나의 결제환경만 선택가능하며, 결제환경 설정은 언제든 변경할 수 있습니다.
* 등록되어 있는 테스트 ID로 어느 환경에서 결제 테스트 할지 설정할 수 있습니다.&#x20;
* 테스트 ID 결제환경을 변경할 경우 반드시 '저장' 버튼을 눌러 새로운 설정을 저장해야 합니다.

### **테스트 ID 삭제하기 (개별 삭제, 일괄 삭제)**

* 테스트 ID가 더 이상 유효하지 않을 경우 테스트 ID를 삭제할 수 있습니다.&#x20;
* '삭제'하기 버튼을 눌러 개별 테스트 ID를 삭제하거나, 여러 테스트 ID를 체크한 후 '체크한 ID 삭제' 버튼을 눌러 일괄 삭제할 수 있습니다.

### **테스트 ID 설정 복사**

* 동일한 테스트 ID를 여러 앱에 공통으로 사용할 경우, 테스트 ID 설정을 다른 앱으로 복사할 수 있습니다.
* 복사가 가능한 앱만 체크 박스가 활성화 되며, 테스트 ID 설정을 복사할 대상 앱을 선택한 수 '복사하기'를 누르면 됩니다.
* 대상 앱에 바이너리(apk)가 등록되어 있지 않을 경우 Sandbox로 설정된 테스트 ID만 복사됩니다.

## **Sandbox 환경에서의 결제 테스트 (필수)**  <a href="#id-03.-sandbox" id="id-03.-sandbox"></a>

### Sandbox 결제 테스트 및 검증  <a href="#id-03.-sandbox" id="id-03.-sandbox"></a>

Sandbox는 상용 결제환경이 아닌 가상의 결제환경으로, 결제화면에서 결제실패 또는 성공 중 원하는 응답을 선택하면 해당 응답결과를 전송합니다.

Sandbox에서 결제한 내역은 인앱결제 테스트 화면의 'Sandbox' 탭에서 조회할 수 있으며, 결제취소도 가능합니다.

OS 별로 Sandbox에서 테스트를 1회 이상 진행해야 하며, 테스트를 수행하지 않은 경우에는 검증을 요청 할 수 없습니다.&#x20;

> 국가/지역 별 테스트는 필요하지 않습니다. 테스트하기 편한 국가/지역에서 테스트하시면 됩니다.&#x20;

원게임루프를 통해  AAB/APK를 배포하는 경우에는 안드로이드 OS 테스트 결과를 참조하므로, 안드로이드 환경에서 테스트를 진행하시면 됩니다.&#x20;

테스트 환경 (상용 및 Sandbox) 정보는 원스토어 서비스와 연결된 동안 메모리에 유지됩니다. 만약 개발자 센터를 통해 테스트 ID의 테스트 환경을 변경했다면, 반드시 앱을 종료하고 다시 시작해야 합니다.

{% hint style="info" %}
참고사항

* 테스트 ID 설정이 Sandbox로 되어 있어야 Sandbox 결제 테스트가 가능합니다.
{% endhint %}

### 구독형 상품의 Sandbox 테스트  <a href="#id-03.-sandbox" id="id-03.-sandbox"></a>

구독형 상품을 개발할 때 실제 시간과 동일하게 결제 주기가 설정된다면, 테스트에 너무 많은 시간이 필요합니다.&#x20;

이에 Sandbox 환경에서는 구독 주기별 소요 시간을 조정하여 테스트를 원활하게 할 수 있도록 지원하고 있습니다.&#x20;

각 결제 주기 및 기능 별로 Sandbox 내 시간 흐름은 다음과 같습니다.&#x20;

| 결제 주기 | Sandbox 내 시간 |
| ----- | ------------ |
| 1주    | 5분           |
| 1개월   | 5분           |
| 3개월   | 10분          |
| 6개월   | 15분          |
| 1년    | 30분          |

| 기능                | Sandbox 내 시간 |
| ----------------- | ------------ |
| 무료 구독             | 3분           |
| 유예 기간             | 5분           |
| 계정 보류             | 10분          |
| 가격 변경 반영 (7일)     | 5분           |
| 가격 변경 동의 기간 (30일) | 5분           |

## **상용테스트 환경에서의 결제 테스트 (선택)**  <a href="#id-03." id="id-03."></a>

상용테스트는 상용 결제환경에서 해당 인앱상품의 결제를 진행하여 결제 결과를 전송합니다.

상용테스트로 결제한 내역은 인앱결제 테스트 화면의 '상용테스트' 탭에서 조회할 수 있으며, 결제취소도 가능합니다.

상용테스트 환경에서 결제를 하실 경우 취소하지 않으면 과금이 발생할 수 있으니, 결제 테스트 후 반드시 결제취소를 진행해주셔야 합니다.

{% hint style="info" %}
참고사항

* 테스트 ID 설정이 상용테스트로 되어 있어야 상용테스트 환경에서 결제 테스트가 가능합니다.
* 상용테스트로 설정하지 않은 테스트 ID 또는 테스트 ID로 등록되어있지 않은 원스토어 ID로 결제할 경우 테스트 상용결제가 아닌 실제 상용결제가 진행되어 과금되니 주의해주시기 바랍니다.&#x20;
* 실제 상용결제가 진행된 경우, 즉시 원스토어 측으로 결제정보를 전달하여 결제취소를 요청하시기 바랍니다.
{% endhint %}

## **인앱결제 테스트 결과 확인 및 결제취소** <a href="#id-03." id="id-03."></a>

인앱결제 테스트 내역의 결제상태 확인 및 결제취소를 할 수 있습니다.&#x20;

Sandbox에서 결제한 내역은 인앱결제 테스트 화면의 'Sandbox' 탭에서, 상용테스트로 결제한 내역은 인앱결제 테스트 화면의 '상용테스트' 탭에서 결제내역을 조회할 수 있습니다.

*   #### Sandbox 환경 <a href="#id-03.-sandbox" id="id-03.-sandbox"></a>

    * Sandbox 환경에서 결제 테스트한 내역이 제공됩니다.&#x20;
    * 결제한 상품정보 및 결제상태를 조회할 수 있으며, 결제취소도 가능합니다.


* #### **상용테스트 환경** <a href="#id-03." id="id-03."></a>
  * 상용테스트 환경에서 결제 테스트한 내역이 제공됩니다.
  * 결제한 상품정보 및 결제상태를 조회할 수 있으며, 결제취소도 가능합니다.

## **보안 및 인증** <a href="#id-03." id="id-03."></a>

### **인앱 결제의 보안** <a href="#id-03." id="id-03."></a>

*   원스토어 인앱 결제는 데이터의 위변조 여부를 확인하는 방법으로 서명(Signature)의 유효성 체크 방식을 사용합니다.

    모바일 앱은 많은 공격의 위협에 노출되어 있으므로 이러한 위험을 최소화하기 위해 개발사 앱이나 서버에서 서명 인증을 진행하는 것을 권장합니다.&#x20;

    다음은 좀 더 안전한 인앱결제를 위해 개발사에서 추가로 준비해야 할 사항입니다.
* 개발사 서버 사용
  * APK Reverse Engineering 등을 이용한 공격을 어렵게 하기 위해서 구현 코드와 공개 키(Public key) 등은 서버에서 보관하고 인증을 진행할 수 있도록 합니다.
  * 구매 정보를 저장할 때도 단말기의 저장소보다 개발사 서버의 저장소를 사용하여 아이템의 사용 허가를 확인하도록 하는 것이 좋습니다.
* 변경된 코드 사용
  * 원스토어에서 제공하는 샘플 코드는 많은 사람들에게 오픈 되어 있는 코드로, 원본을 그대로 사용하는 것보다는 수정해서 사용하는 것을 권장합니다. 동일한 코드를 사용하면 그만큼 공격의 위험에 노출되기 쉽습니다.
  * Proguard 같은 코드 난독화 툴을 이용하여 결제 관련 코드를 보호하는 것이 좋습니다.
  * 공개 키라 할지라도 앱 코드 안에 일반 문자열로 넣는 것은 안전한 방법이 아닙니다. 다른 문자열과 XOR하여 쉽게 노출되지 않도록 하는 등 공격자가 쉽게 접근 할 수 없도록 하는 것이 안전합니다.
* 결제 요청 시 ‘developerPayload’ 필드를 활용
  *   개발사는 결제 요청 시 ‘developerPayload’ 필드에 임의의 정보를 넣고 결제 완료 시 다시 전달 받을 수 있습니다.&#x20;

      해당 필드에 timestamp 등을 조합한 추가적인 보안 검증용 데이터를 넣어 결제 결과에 대한 검증을 좀 더 안전하게 할 수 있습니다.

### **원스토어 인앱결제 인증 방법** <a href="#id-03." id="id-03."></a>

* 인증 준비
  *   인증을 위한 키 알고리즘은 RSA 방식을 사용하며, 서명을 위한 알고리즘으로 ‘SHA512withRSA’를 사용합니다.

      서명 검증키는 [사전준비](pre) 페이지 내 '라이선스 키 및 OAuth 인증 정보 확인' 부분을 참고하시기 바랍니다.

      서명 검증키의 값은 단말기나 서버 등 인증을 하는 위치에 따라 적절하게 저장하여 사용합니다.
* 서명 인증 샘플 코드
  *   SDK에서는 'AppSecurity' 유틸리티 클래스를 제공하며 'verifyPurchase' 메서드를 이용해 서명 인증을 할 수 있습니다.&#x20;

      SDK를 이용하지 않고 개발사가 직접 서명 인증을 하려면 'AppSecurity'와 동일한 기능을 하는 코드를 직접 구현해야 합니다.&#x20;

      자세한 구현 내용은 SDK 라이브러리와 함께 배포하는 샘플을 참고합니다.



출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/sdk.md
# 04. 원스토어 인앱결제 SDK를 사용해 구현하기

## 개요

원스토어 결제 라이브러리는 Java 환경에서 최신 기능을 제공합니다.\
이 가이드에서는 원스토어 결제 라이브러리 기능 구현하는 방법을 설명합니다.

## 프로젝트 설정

### 레파지토리 및 종속 항목 추가

프로젝트 최상위 `gradle` 파일에 원스토어 `maven` 주소를 등록합니다.

{% hint style="info" %}
Android Studio (version: bumblebee) 이상에서는 `settings.gradle`파일에서 추가합니다.
{% endhint %}

```gradle
repositories {
    ... 
    maven { url 'https://repo.onestore.net/repository/onestore-sdk-public' }
}
```

앱의 `build.gradle` 파일에 원스토어 결제 라이브러리 종속 항목을 추가합니다.

```gradle
dependencies {
    ...
    def iap_lastest_version = "21.xx.xx"
    implementation "com.onestorecorp.sdk:sdk-iap:$iap_lastest_version"
}
```

### \<queries> 설정

`AndroidManifest.xml` 파일에 `<queries>`를 설정 해야합니다.\
자세한 내용은 [공지사항](https://dev.onestore.co.kr/devpoc/support/news/noticeView.omp?noticeId=32968)을 참조하세요.

{% hint style="danger" %}
**\<queries>** 태그를 설정하지 않으면 SDK에서 원스토어 서비스를 찾을 수 없습니다.
{% endhint %}

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

### 스토어 지정을 위한 개발자 옵션 설정

#### v21.02.00 업데이트 – 글로벌 스토어 선택 기능 추가

&#x20;IAP SDK 21.02.00 부터 아래와 같이 `onestore:dev_option`의 `android:value` 값을 설정하면, SDK와 연동되는 스토어 앱을 지정 할 수 있습니다.

```xml
<manifest>
    <application>
        <meta-data android:name="onestore:dev_option" android:value="onestore_01" />
    </application>
</manifest>
```

<table><thead><tr><th width="226">값 (android:value)</th><th>적용 대상 국가</th></tr></thead><tbody><tr><td><code>onestore_00</code></td><td>대한민국 (South Korea) <em>(기본값)</em></td></tr><tr><td><code>onestore_01</code></td><td>싱가포르, 타이완 (Singapore, Taiwan)</td></tr><tr><td><code>onestore_02</code></td><td>미국 – Digital Turbine (United States)</td></tr></tbody></table>

{% hint style="warning" %}
21.01.00 버전에서는 android:value 값이 global만 설정 가능하며, 싱가포르/타이완 스토어 앱만 지정이 가능했습니다.&#x20;
{% endhint %}

{% hint style="danger" %}
**주의**: 배포 버전의 바이너리에서는 이 옵션을 반드시 제거해주세요.&#x20;
{% endhint %}



## 인앱 라이브러리 적용하기

### 로그레벨 설정

개발 단계에서 로그 레벨을 설정하여 SDK의 데이터의 흐름을 좀 더 자세히 노출할 수 있습니다.\
`android.util.Log`에 정의된 값을 기반으로 동작합니다.

```kotlin
/**
 * Set the log level.<br/>
 * {@link Log#VERBOSE}, {@link Log#DEBUG}, {@link Log#INFO},
 * {@link Log#WARN}, {@link Log#ERROR}
 * @param level int
 */
com.gaa.sdk.base.Logger.setLogLevel(2)
```

| 상수             | 값 |
| -------------- | - |
| VERBOSE        | 2 |
| DEBUG          | 3 |
| INFO (default) | 4 |
| WARN           | 5 |
| ERROR          | 6 |

{% hint style="danger" %}
배포 빌드 버전에서는 보안에 취약할 수 있으니 이 옵션을 **삭제**해야 합니다.
{% endhint %}

### 오류 처리 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

원스토어 결제 라이브러리는 `IapResult` 형식으로 오류를 반환합니다. `IapResult`에는 앱에서 발생할 수 있는 결제 관련 오류를 분류하는 `ResponseCode`가 포함되어 있습니다. 예를 들어 `RESULT_NEED_LOGIN` 이나 `RESULT_NEED_UPDATE` 같은 오류 코드가 수신되면 앱에서 그에 맞는 처리를 해야 합니다.

### 원스토어 로그인 하기 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

**`GaaSignInClient` 초기화**

`GaaSignInClient`는 원스토어 로그인을 하기 위한 라이브러리입니다.\
`getClient()` 통해 인스턴스를 생성합니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
val signInClient = GaaSignInClient.getClient(activity)
```
{% endtab %}

{% tab title="Java" %}
```java
GaaSignInClient signInClient = GaaSignInClient.getClient(activity);
```
{% endtab %}
{% endtabs %}

**백그라운드 로그인하기**

`slientSignIn()` 통해 백그라운도 로그인을 호출합니다.

사용자가 이미 원스토어 계정에 로그인 되어있을 경우 이후 부터는 백그라운드에서 토큰로그인을 시도합니다. 성공 또는 실패에 대한 결과값으로 `SignInResult` 객체로 응답을 받습니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
signInClient.silentSignIn { signInResult ->
  
}
```
{% endtab %}

{% tab title="Java" %}
```java
signInClient.silentSignIn(new OnAuthListener() {
    @Override
    public void onResponse(@NonNull SignInResult signInResult) {
        
    }
});
```
{% endtab %}
{% endtabs %}

**포그라운드로** **로그인하기**

`slientSignIn()`과 달리 해당 함수는 `UiThread` 에서만 호출해야 합니다.

기본적으로 먼저 백그라운드 로그인을 시도하지만 실패에 대한 처리를 SDK에서 전담하여 처리합니다.\
이 후, 로그인 화면을 띄워 사용자에게 로그인을 유도합니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
signInClient.launchSignInFlow(activity) { signInResult ->
  
}
```
{% endtab %}

{% tab title="Java" %}
```java
signInClient.launchSignInFlow(activity, new OnAuthListener() {
    @Override
    public void onResponse(@NonNull SignInResult signInResult) {
        
    }
});
```
{% endtab %}
{% endtabs %}

### PurchaseClient 초기화 <a href="#id-04.-sdk-purchaseclient" id="id-04.-sdk-purchaseclient"></a>

`PurchaseClient`는 원스토어 결제 라이브러리와 앱 간의 통신을 위한 기본 인터페이스입니다.

단일 이벤트에 관한 여러 개의 `PurchasesUpdatedListener` 콜백이 발생하는 상황을 피할 수 있도록 `PurchaseClient` 연결을 열어 두는 것을 권장합니다.

`PurchaseClient`를 생성하려면 `newBuilder()`를 사용합니다. 구매 관련 업데이트를 수신하려면 `setListener()`를 호출하여 `PurchasesUpdatedListener`에 대한 참조를 전달해야 합니다. 이 리스너는 앱의 모든 구매 관련 업데이트를 수신합니다.\
`setBase64PublicKey()`는 SDK에서 구매 데이터 위변조에 대한 서명 확인 작업을 합니다. 옵션 값이지만 사용하는 것을 권장합니다.

라이선스 키는 '라이선스 관리' 메뉴에서 확인 가능합니다.

{% hint style="success" %}
라이선스 키의 경우, 앱 내 코드로 저장하기 보다는 보안이 될 수 있도록 서버 등을 이용하여 전달받아 사용하는 것을 권장합니다.
{% endhint %}

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
private val listener = PurchasesUpdatedListener { iapResult, purchases ->
       // To be implemented in a later section.
}

private var purchaseClient = PurchaseClient.newBuilder(activity)
   .setListener(listener)
   .setBase64PublicKey(/*your public key*/) // optional
   .build()
```
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

### 원스토어 서비스 연결 설정 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

`PurchaseClient`를 생성 후 원스토어 서비스에 연결해야 합니다.

연결하려면 `startConnection()`을 호출합니다. 연결 프로세스는 비동기이며 클라이언트 연결이 완료되면 `PurchaseClientStateListener`를 통해 콜백이 수신됩니다.

다음 예는 연결을 시작하고 사용 준비가 되었는지 테스트하는 방법입니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
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
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

### 상품 상세정보 조회하기 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

상품 상세정보를 조회하려면 `queryProductDetailsAsync()`를 호출합니다. 사용자에게 제품을 표시하기 전에 중요한 단계입니다.

`queryProductDetailsAsync()`를 호출할 때 `setProductType()`과 함께 원스토어 개발자 센터에서 생성된 인앱 상품 ID 문자열 목록을 지정하는 `ProductDetailParams`의 인스턴스를 전달합니다.

`ProductType`은 아래와 같습니다.

<table><thead><tr><th width="246">Product</th><th>Enum</th></tr></thead><tbody><tr><td>관리형 상품</td><td><code>ProductType.INAPP</code></td></tr><tr><td>정기 결제 상품 (구독 상품)</td><td><code>ProductType.SUBS</code></td></tr><tr><td><del>월 정액 상품</del></td><td><del><code>ProductType.AUTO</code></del> (이 상품은 향후 지원되지 않을 예정입니다.) </td></tr></tbody></table>

위의 모든 유형의 데이터를 한 번에 조회하고 싶을 경우는 `ProductType.ALL` 설정하면 됩니다.

{% hint style="warning" %}
`ProductType.ALL`은 [상품 상세정보 조회하기](#id-04.-sdk-2)에서만 사용할 수 있으며, [구매 요청하기](#id-04.-sdk-3), [구매 내역 조회하기](#id-04.-sdk-8) 에서는 사용할 수 없습니다.
{% endhint %}

비동기 작업 결과를 처리하려면 `ProductDetailsListener` 인터페이스를 구현해야 합니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
val params = ProductDetailsParams.newBuilder()
        .setProductIdList(productIdList)
        .setProductType(ProductType.INAPP)
        .build()
purchaseClient.queryProductDetailsAsync(params) { iapResult, productDetails -> 
    // Process the result.
}
```
{% endtab %}

{% tab title="Java" %}
{% code fullWidth="false" %}
```java
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
{% endcode %}
{% endtab %}
{% endtabs %}

### 구매 요청하기 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

앱에서 구매 요청을 하기 위해서는 기본 스레드에서 `launchPurchaseFlow()` 함수를 호출합니다.

이 함수는 `queryProductDetailsAsync()` 함수를 호출해서 얻은 `ProductDetail` 객체의 값을 토대로 `PurchaseFlowParams` 객체를 생성합니다.\
`PurchaseFlowParams` 객체를 생성하려면 `PurchaseFlowParams.Builder` 클래스를 사용합니다.

`setDeveloperPayload()`는 개발사에서 임의로 입력한 값으로 최대 200byte입니다. 이 값은 결제 후에 데이터의 정합성과 부가 데이터를 확인하기 위해 사용할 수 있습니다.\
`setProductName()`은 상품 이름을 결제 시 변경하여 노출하고 싶을 때 사용됩니다.\
`setQuantity()`는 관리형 인앱 상품에만 적용되며 한 상품을 여러 개 구매할 때 사용됩니다.

{% hint style="success" %}
원스토어는 사용자에게 할인 쿠폰, 캐시백 등의 다양한 혜택 프로모션을 진행하고 있습니다.\
개발사는 구매 요청 시에 `gameUserId`, `promotionApplicable` 파라미터를 이용하여 앱을 사용하는 유저의 프로모션 참여를 제한하거나 허용할 수 있습니다.\
개발사는 앱의 고유한 유저 식별 번호 및 프로모션 참여 여부를 선택하여 전달하고, 원스토어는 이 값을 토대로 사용자의 프로모션 혜택을 적용하게 됩니다.
{% endhint %}

{% hint style="warning" %}
`gameUserId`, `promotionApplicable` 파라미터는 옵션 값으로 원스토어 사업 부서 담당자와 프로모션에 대해 사전협의가 된 상태에서만 사용하여야 하며, 일반적인 경우에는 값을 보내지 않습니다.\
또한, 사전협의가 되어 값을 보낼 경우에도 개인 정보보호를 위해 `gameUserId`는 _has&#x68;_&#xB41C; 고유한 값으로 전송하여야 합니다.
{% endhint %}

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
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
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

`launchPurchaseFlow()` 호출에 성공하면 아래와 같은 화면을 표시합니다. \[그림 1]은 정기 결제 구매 화면을 나타냅니다.

<figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2F2Lkq8XusttkJ8sImrM8C%2F2.png?alt=media&#x26;token=9da3eb3f-02b1-45f9-9727-2c200068c65d" alt=""><figcaption><p>그림 1</p></figcaption></figure>

구매에 성공하면 `PurchasesUpdatedListener` 인터페이스의 `onPurchasesUpdated()` 함수에 구매 작업 결과가 전송됩니다. 이 리스너는 [PurchaseClient 초기화](#id-04.-sdk-purchaseclient) 할 때 `setListener()` 함수를 사용하여 지정됩니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
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
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

<figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2F58l823PfVJNsedj5q9sw%2F3.png?alt=media&#x26;token=f7c93cc2-cc7b-482d-af7f-3b491fb92cd8" alt=""><figcaption><p>그림 2</p></figcaption></figure>

구매가 성공하면 구매 데이터에는 사용자 및 상품 ID를 나타내는 고유 식별자인 구매 토큰도 생성됩니다. 구매 토큰을 앱 내에 저장할 수 있지만 구매를 인증하고 사기로부터 보호할 수 있는 백엔드 서버로 토큰을 전달하는 것이 좋습니다.

관리형 상품과 정기 결제 상품의 구매 토큰은 결제가 일어날 때마다 구매 토큰이 발행됩니다. (월정액 상품의 경우 자동 결제가 갱신되는 동안 구매 토큰은 동일하게 유지됩니다.)

또한 사용자는 영수증 번호가 포함된 거래 영수증을 이메일로 받습니다. 관리형 상품은 구매할 때마다 이메일을 받으며, 월정액 상품과 정기 결제 상품은 처음 구매 시 그리고 이후 갱신될 때마다 이메일을 받습니다.

### 정기 결제 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

정기 결제는 취소될 때까지 자동으로 갱신됩니다. 정기 결제는 다음 상태를 가질 수 있습니다.

* **활성**: 사용자가 콘텐츠 사용에 문제가 없는 양호한 상태이며 정기 결제에 접근할 수 있습니다.
* **일시 중지 예약**: 사용자가 정기 결제를 이용 중 일시 중지를 하고 싶을 때 선택할 수 있습니다.
  * 주간 정기 결제: 1\~3주 단위로 일시 중지할 수 있습니다.
  * 월간 정기 결제: 1\~3개월 단위로 일시 중지할 수 있습니다.
  * 연간 정기 결제: 일시 중지를 지원하지 않습니다.
* **해지 예약**: 사용자가 정기 결제를 이용 중이지만 취소하고 싶을 때 선택할 수 있습니다. 다음 결제일에 결제가 되지 않습니다.
* **유예, 보류**: 사용자에게 결제 문제가 발생하면 다음 결제일에 결제가 되지 않습니다. 취소 예약을 할 수 없으며 즉시 "구독 해지"를 할 수 있습니다.

#### 사용자가 정기 결제를 업그레이드, 다운그레이드 또는 변경할 수 있도록 허용 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

사용자가 정기 결제를 업그레이드하거나 다운그레이드 하려면 구매 시 _비례 배분 모&#xB4DC;_&#xB97C; 설정하거나 변경사항이 정기 결제 사용자에게 영향을 주는 방식을 설정할 수 있습니다.

<figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2FeR3I8k472b8zq7EV30Di%2F4.png?alt=media&#x26;token=ede4417c-24e1-46a3-bb89-8a019cebb045" alt=""><figcaption><p>그림 3</p></figcaption></figure>

다음 표에는 사용 가능한 비례 배분 모&#xB4DC;_(_`PurchaseFlowParams.ProrationMode`_)_&#xAC00; 나와 있습니다.

| 비례 배분 모드                                | 설명                                                                                          |
| --------------------------------------- | ------------------------------------------------------------------------------------------- |
| IMMEDIATE\_WITH\_TIME\_PRORATION        | 정기 결제의 교체가 즉시 이루어지며, 남은 시간은 가격 차이를 기반으로 조정되어 입금되거나 청구됩니다. (이것은 기본 동작입니다.)                   |
| IMMEDIATE\_AND\_CHARGE\_PRORATED\_PRICE | 정기 결제의 교체가 즉시 이루어지며, 청구 주기는 동일하게 유지됩니다. 나머지 기간에 대한 가격이 청구됩니다. (이 옵션은 업그레이드 에서만 사용할 수 있습니다.) |
| IMMEDIATE\_WITHOUT\_PRORATION           | 정기 결제의 교체가 즉시 이루어지며, 다음 결제일에 새로운 가격이 청구됩니다. 청구 주기는 동일하게 적용됩니다.                              |
| DEFERRED                                | 기존 요금제가 만료되면 교체가 적용되며 새 요금이 동시에 청구됩니다.                                                      |

#### 업그레이드 또는 다운그레이드 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

정기 결제는 [구매 요청하기](#id-04.-sdk-3)와 동일한 API를 사용하여 사용자에게 업그레이드 또는 다운그레이드를 제공할 수 있습니다. 다만, 정기 결제의 업그레이드 다운그레이드를 적용하기 위해선 기존 정기 결제 구매 토큰과 비례 배분 모드 값이 필수로 필요합니다.

다음 예와 같이 현재 정기 결제, 향후(업그레이드 또는 다운그레이드) 정기 결제 및 비례 배분 모드에 관한 정보를 제공해야 합니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
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
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

업그레이드 또는 다운그레이드의 경우도 [구매 요청하기](#id-04.-sdk-3) 로직을 수행하기 때문에 응답은 `PurchasesUpdatedListener`에서 수신합니다. 또한 [구매 내역 조회하기](#id-04.-sdk-8)에서도 요청 시 응답을 받을 수 있습니다.\
비례 배분 모드로 구매했을 때도 일반 구매와 동일하게 `PurchaseClient.acknowledgeAsync()`를 사용하여 _구매 확인을_ 해야 합니다.

### 구매 처리하기  <a href="#id-04.-sdk" id="id-04.-sdk"></a>

구매가 완료되면 앱에서 구매 확인 처리를 해야 합니다. 대부분의 경우 앱은 `PurchasesUpdatedListener`를 통해 구매 알림을 받습니다.\
또는  [구매 내역 조회하기](#id-04.-sdk-8)에서 설명된 것처럼 앱이 `PurchaseClient.queryPurchasesAsync()` 함수를 호출하여 처리하는 경우가 있습니다.

다음 메서드 중 하나를 사용해 구매를 확인할 수 있습니다.

* 소모성 제품인 경우 `PurchaseClient.consumeAsync()`를 사용합니다.
* 소모성 제품이 아니라면 `PurchaseClient.acknowledgeAsync()`를 사용합니다.

#### 관리형 상품 소비하기 (consume) <a href="#id-04.-sdk-consume" id="id-04.-sdk-consume"></a>

관리형 상품은 소비를 하기 전까지는 재구매 할 수 없습니다.

상품을 소비하기 위해서는 `consumeAsync()`를 호출합니다. 또한 소비 작업 결과를 전달받으려면 `ConsumeListener` 인터페이스를 구현해야 합니다.

{% hint style="info" %}
관리형 상품을 소비하지 않으면 영구성 형태의 상품 타입처럼 활용할 수 있으며, 구매 후 즉시 소비하면 소비성 형태의 상품으로도 활용됩니다.\
또한 특정 기간 이후에 소비하면 기간제 형태의 상품으로 활용할 수 있습니다.
{% endhint %}

{% hint style="danger" %}
3일 이내에 구매를 확인(`acknowledge`) 또는 소비(`consume`)를 하지 않으면 사용자에게 상품이 지급되지 않았다고 판단되어 자동으로 환불됩니다.
{% endhint %}

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
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
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

{% hint style="info" %}
소비 요청이 때로 실패할 수 있으므로 보안 백엔드 서버를 확인하여 각 구매 토큰이 사용되지 않았는지 확인해야 합니다. 그래야 앱이 동일한 구매에 대해 여러 번 자격을 부여하지 않습니다. 또는 자격을 부여하기 전에 성공적인 소비 응답을 받을 때까지 기다릴 수 있습니다.
{% endhint %}

#### 구매 확인하기 (acknowledge) <a href="#id-04.-sdk-acknowledge" id="id-04.-sdk-acknowledge"></a>

비 소비성 상품에 대해 구매 확인 처리를 하려면 `PurchaseClient.acknowledgeAsync()` 함수를 사용합니다. 관리형 상품,  월정액 상품, 구독형 상품 모두 사용할 수 있습니다.

`PurchaseData.isAcknowledged()` 함수를 사용하여 구매 확인 되었는지를 판단 할 수 있습니다. 또한 구매 확인에 대한 작업 결과를 전달받으려면 `AcknowledgeListener` 인터페이스를 구현해야 합니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
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
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

{% hint style="warning" %}
`AcknowledgeListener.onAcknowledgeResponse()` 함수로 전달된 `PurchaseData`는 요청할 당시의 데이터이기 때문에 `acknowledgeState` 값이 변경되지 않습니다. [구매 내역 조회하기](#id-04.-sdk-8)를 통해 변경된 데이터로 교체해야 합니다.
{% endhint %}

### 구매 내역 조회하기 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

`PurchasesUpdatedListener`를 사용하여 구매를 처리하는 것만으로는 모든 구매가 처리 되었다는 것을 보장할 수 없습니다. 앱에서 구매 추적을 놓치거나 구매를 인식하지 못할 수 있는 몇 가지 시나리오는 다음과 같습니다.

아래와 같은 시나리오에서 앱은 구매 응답을 받지 못하거나 구매를 인식하지 못하는 경우가 발생할 수 있습니다.&#x20;

* **네트워크 문제**: 사용자가 성공적으로 구매를 하였고, 원스토어에서 확인도 받았지만 기기가 _PurchasesUpdatedListene&#x72;_&#xB97C; 통해 구매 알림을 받기 전에 네트워크 연결이 끊어졌을 경우
* **여러 기기**: 한 기기에서 항목을 구입한 후 다른 기기로 전환하는 경우&#x20;

이러한 경우를 대비하여 앱의 `onCreate()`_나_ `onResume()`에서 `PurchaseClient.queryPurchasesAsync()`를 호출하여 구매가 성공적으로 처리 되었는지 확인해야 합니다.

콜백 처리는 `PurchasesUpdatedListener` 와 동일합니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
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
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

### 월정액 상품 상태 변경하기 (Deprecated) <a href="#id-04.-sdk-deprecated" id="id-04.-sdk-deprecated"></a>

월정액 상품은 최초 구매 후 익월 동일일에 갱신이 이루어지는 상품입니다. 월정액 상품의 상태는 `PurchaseData.getRecurringState()`를 통해 확인할 수 있습니다.

월정액 상품의 상태를 변경하려면 `PurchaseClient.manageRecurringProductAsync()`를 사용합니다.\
`RecurringProductParams` 객체에 구매 데이터와 변경하려는 `PurchaseClient.RecurringAction` 값을 입력합니다.&#x20;

SDK V21 (API V7) 부터는 새로운 월정액 상품을 만들 수 없습니다. 결제 주기가 한 달인 구독형 상품을 이용하세요.&#x20;

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
// Purchase retrieved from PurchaseClient#queryPurchasesAsync or your PurchasesUpdatedListener.
fun manageRecurring(purchase: PurchaseData) {
    val recurringParams = RecurringProductParams.newBuilder()
            .setPurchaseData(purchase)
            .setRecurringAction(RecurringAction.CANCEL | RecurringAction.REACTIVATE)
            .build()

    purchaseClient.manageRecurringProductAsync(recurringParams) { iapResult, purchaseData, action ->
        // PurchaseClient by calling the queryPurchasesAsync() method.
    }
}
```
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

{% hint style="warning" %}
`RecurringProductListener.onRecurringResponse()` 함수로 전달된 `PurchaseData`는 요청할 당시의 데이터이기 때문에 `recurringState` 값이 변경되지 않습니다. [구매 내역 조회하기](#id-04.-sdk-8)를 통해 변경된 데이터로 교체해야 합니다.
{% endhint %}

### 정기 결제 관리 화면을 열기 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

정기 결제 중인 상품을 관리하는 화면을 띄울 수 있습니다.

매개변수로 `SubscriptionsParams`에 `PurchaseData`를 포함해서 넣으면 구매 데이터를 확인하여 해당 정기 결제 상품의 관리 화면을 실행합니다.\
그러나 `SubscriptionParams`을 _`null`_ 넣을 경우 사용자의 정기 결제 리스트 화면을 실행합니다.

다음은 정기 결제 관리 화면을 띄우는 방법을 나타내는 예제입니다.

{% tabs %}
{% tab title="Kotlin" %}
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
{% endtab %}

{% tab title="Java" %}
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
{% endtab %}
{% endtabs %}

### 마켓 구분 코드 얻기 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

SDK v19 이상부터 Server API를 사용하기 위해서는 마켓 구분 코드가 필요합니다.\
`getStoreInfoAsync()`를 통해서 마켓 구분 코드를 획득할 수 있습니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
purchaseClient.getStoreInfoAsync { iapResult, storeCode ->
    // Save storecode and use it in Server to Server API.
}
```
{% endtab %}

{% tab title="Java" %}
```java
purchaseClient.getStoreInfoAsync(new StoreInfoListener() {
    @Override
    public void onStoreInfoResponse(IapResult iapResult, String storeCode) {
        // Save storecode and use it in Server to Server API.
    }
});
```
{% endtab %}
{% endtabs %}

### StoreEnvironment API 기능 추가

`StoreEnvironment.getStoreType()` API는 SDK가 탑재된 애플리케이션이 원스토어를 통해 설치되었는지를 판단하는 기능을 제공합니다.

#### Store Type 정의

해당 API는 `StoreType`을 반환하며, 아래 네 가지 값 중 하나를 가집니다.

<table><thead><tr><th width="240">StoreType</th><th width="71">value</th><th>description</th></tr></thead><tbody><tr><td><code>StoreType.UNKNOWN</code></td><td>0</td><td>앱 설치 스토어 정보를 알 수 없음 <em>(APK 직접 설치, 출처 불명 등)</em></td></tr><tr><td><code>StoreType.ONESTORE</code></td><td>1</td><td>ONE Store에서 설치됨 <em>(또는 개발자 옵션이 활성화된 경우)</em></td></tr><tr><td><code>StoreType.VENDING</code></td><td>2</td><td>Google Play Store에서 설치됨</td></tr><tr><td><code>StoreType.ETC</code></td><td>3</td><td>기타 스토어에서 설치됨</td></tr></tbody></table>

#### API 사용 방법

해당 API는 `StoreEnvironment.getStoreType()`을 호출하여 사용할 수 있습니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
import com.gaa.sdk.base.StoreEnvironment

val storeType = StoreEnvironment.getStoreType()

when (storeType) {
    StoreType.ONESTORE -> println("ONE Store에서 설치된 앱입니다.")
    StoreType.VENDING -> println("Google Play Store에서 설치된 앱입니다.")
    StoreType.ETC -> println("기타 스토어에서 설치된 앱입니다.")
    StoreType.UNKNOWN -> println("스토어 정보를 알 수 없습니다.")
}
```
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

#### 스토어 판단 기준

이 API는 세 가지 방법을 통해 설치된 스토어를 판별합니다.

1. &#x20;원스토어 마켓 서명을 통해 배포된 경우
   1. 원스토어의 마켓 서명을 통한 배포 여부를 확인하여, 원스토어에서 설치된 앱인지 확인합니다.
2. Installer Package Name을 기반으로 판별
   1. 원스토어의 마켓 서명을 통해 배포되지 않은 경우, PackageManager.getInstallerPackageName() API를 이용하여 앱 설치 시 사용된 스토어 정보를 확인합니다.
3. 개발자 옵션(`onestore:dev_option`)이 활성화된 경우
   1. onestore:dev\_option 이 설정 되어있으면 무조건 StoreType.ONESTORE로 응답합니다.

#### 활용 예시

스토어별 UI 차별화 적용

원스토어와 다른 앱 마켓에서 제공하는 결제 시스템이 다를 경우, UI를 다르게 설정할 수 있습니다.

```kotlin
if (StoreEnvironment.getStoreType() == StoreType.ONESTORE) {
    showOneStorePaymentUI()
} else {
    showDefaultPaymentUI()
}
```

스토어별 기능 차단

특정 기능을 원스토어에서만 사용하도록 설정할 수 있습니다.

```kotlin
if (StoreEnvironment.getStoreType() != StoreType.ONESTORE) {
    println("이 기능은 ONE Store에서만 사용할 수 있습니다.")
    return
}
enableOneStoreExclusiveFeature()
```

### 원스토어 서비스 설치하기 <a href="#id-04.-sdk" id="id-04.-sdk"></a>

원스토어 서비스의 버전이 낮거나 없을 경우 인앱결제를 이용할 수 없습니다. `PurchaseClient.startConnection()`을 통해 연결할 때 `IapResult.getResponseCode()`에서 확인할 수 있습니다.\
`RESULT_NEED_UPDATE`가 발생하면 `launchUpdateOrInstallFlow()` 메서드를 호출해야 합니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
val activity: Activity = ...

purchaseClient.launchUpdateOrInstallFlow(activity) { iapResult ->
    if (iapResult.isSuccess) {
        // If the installation is completed successfully,
        // you should try to reconnect with the ONE store service. 
        // PurchaseClient by calling the startConnection() method.
    }
}
```
{% endtab %}
{% endtabs %}


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references.md
# 05. 원스토어 인앱결제 레퍼런스

- [Annotations](/dev/tools/billing/v21/references/annotations.md)
- [PurchaseClient.ConnectionState](/dev/tools/billing/v21/references/annotations/purchaseclient.connectionstate.md)
- [PurchaseClient.FeatureType](/dev/tools/billing/v21/references/annotations/purchaseclient.featuretype.md)
- [PurchaseClient.ProductType](/dev/tools/billing/v21/references/annotations/purchaseclient.producttype.md)
- [PurchaseClient.RecurringAction](/dev/tools/billing/v21/references/annotations/purchaseclient.recurringaction.md)
- [PurchaseClient.ResponseCode](/dev/tools/billing/v21/references/annotations/purchaseclient.responsecode.md)
- [PurchaseData.AcknowledgeState](/dev/tools/billing/v21/references/annotations/purchasedata.acknowledgestate.md)
- [PurchaseData.PurchaseState](/dev/tools/billing/v21/references/annotations/purchasedata.purchasestate.md)
- [PurchaseData.RecurringState](/dev/tools/billing/v21/references/annotations/purchasedata.recurringstate.md)
- [PurchaseFlowParams.ProrationMode](/dev/tools/billing/v21/references/annotations/purchaseflowparams.prorationmode.md)
- [Classes](/dev/tools/billing/v21/references/classes.md)
- [AcknowledgeParams](/dev/tools/billing/v21/references/classes/acknowledgeparams.md)
- [AcknowledgeParams.Builder](/dev/tools/billing/v21/references/classes/acknowledgeparams.builder.md)
- [ConsumeParams](/dev/tools/billing/v21/references/classes/consumeparams.md)
- [ConsumeParams.Builder](/dev/tools/billing/v21/references/classes/consumeparams.builder.md)
- [IapResult](/dev/tools/billing/v21/references/classes/iapresult.md)
- [IapResult.Builder](/dev/tools/billing/v21/references/classes/iapresult.builder.md)
- [ProductDetail](/dev/tools/billing/v21/references/classes/productdetail.md)
- [ProductDetailsParams](/dev/tools/billing/v21/references/classes/productdetailsparams.md)
- [ProductDetailsParams.Builder](/dev/tools/billing/v21/references/classes/productdetailsparams.builder.md)
- [PurchaseClient](/dev/tools/billing/v21/references/classes/purchaseclient.md)
- [PurchaseClient.Builder](/dev/tools/billing/v21/references/classes/purchaseclient.builder.md)
- [PurchaseData](/dev/tools/billing/v21/references/classes/purchasedata.md)
- [PurchaseFlowParams](/dev/tools/billing/v21/references/classes/purchaseflowparams.md)
- [PurchaseFlowParams.Builder](/dev/tools/billing/v21/references/classes/purchaseflowparams.builder.md)
- [RecurringProductParams](/dev/tools/billing/v21/references/classes/recurringproductparams.md)
- [RecurringProductParams.Builder](/dev/tools/billing/v21/references/classes/recurringproductparams.builder.md)
- [SubscriptionParams](/dev/tools/billing/v21/references/classes/subscriptionparams.md)
- [SubscriptionParams.Builder](/dev/tools/billing/v21/references/classes/subscriptionparams.builder.md)
- [SubscriptionUpdateParams](/dev/tools/billing/v21/references/classes/subscriptionupdateparams.md)
- [SubscriptionUpdateParams.Builder](/dev/tools/billing/v21/references/classes/subscriptionupdateparams.builder.md)
- [Interfaces](/dev/tools/billing/v21/references/interfaces.md)
- [AcknowledgeListener](/dev/tools/billing/v21/references/interfaces/acknowledgelistener.md)
- [ConsumeListener](/dev/tools/billing/v21/references/interfaces/consumelistener.md)
- [IapResultListener](/dev/tools/billing/v21/references/interfaces/iapresultlistener.md)
- [ProductDetailsListener](/dev/tools/billing/v21/references/interfaces/productdetailslistener.md)
- [PurchaseClientStateListener](/dev/tools/billing/v21/references/interfaces/purchaseclientstatelistener.md)
- [PurchasesListener (deprecated)](/dev/tools/billing/v21/references/interfaces/purchaseslistener-deprecated.md)
- [PurchasesUpdatedListener](/dev/tools/billing/v21/references/interfaces/purchasesupdatedlistener.md)
- [QueryPurchasesListener](/dev/tools/billing/v21/references/interfaces/querypurchaseslistener.md)
- [RecurringProductListener](/dev/tools/billing/v21/references/interfaces/recurringproductlistener.md)
- [StoreInfoListener](/dev/tools/billing/v21/references/interfaces/storeinfolistener.md)


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/annotations.md
# Annotations

- [PurchaseClient.ConnectionState](/dev/tools/billing/v21/references/annotations/purchaseclient.connectionstate.md)
- [PurchaseClient.FeatureType](/dev/tools/billing/v21/references/annotations/purchaseclient.featuretype.md)
- [PurchaseClient.ProductType](/dev/tools/billing/v21/references/annotations/purchaseclient.producttype.md)
- [PurchaseClient.RecurringAction](/dev/tools/billing/v21/references/annotations/purchaseclient.recurringaction.md)
- [PurchaseClient.ResponseCode](/dev/tools/billing/v21/references/annotations/purchaseclient.responsecode.md)
- [PurchaseData.AcknowledgeState](/dev/tools/billing/v21/references/annotations/purchasedata.acknowledgestate.md)
- [PurchaseData.PurchaseState](/dev/tools/billing/v21/references/annotations/purchasedata.purchasestate.md)
- [PurchaseData.RecurringState](/dev/tools/billing/v21/references/annotations/purchasedata.recurringstate.md)
- [PurchaseFlowParams.ProrationMode](/dev/tools/billing/v21/references/annotations/purchaseflowparams.prorationmode.md)


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/annotations/purchaseclient.connectionstate.md
# PurchaseClient.ConnectionState

```
public static @interface PurchaseClient.ConnectionState
```

```
com.gaa.sdk.iap.PurchaseClient.ConnectionState
```

PurchaseClient 의 연결 상태 값.

## Constants <a href="#id-a-purchaseclient.connectionstate-constants" id="id-a-purchaseclient.connectionstate-constants"></a>

***

### CLOSED <a href="#id-a-purchaseclient.connectionstate-closed" id="id-a-purchaseclient.connectionstate-closed"></a>

```
int CLOSED
```

서비스의 연결이 이미 닫혔으므로 다시 사용하면 안됩니다.

Constant Value: 3

### CONNECTED <a href="#id-a-purchaseclient.connectionstate-connected" id="id-a-purchaseclient.connectionstate-connected"></a>

```
int CONNECTED
```

현재 서비스와 연결된 상태입니다.

Constant Value: 2

### CONNECTING <a href="#id-a-purchaseclient.connectionstate-connecting" id="id-a-purchaseclient.connectionstate-connecting"></a>

```
int CONNECTING
```

현재 서비스와 연결이 진행중인 상태입니다.

Constant Value: 1

### DISCONNECTED <a href="#id-a-purchaseclient.connectionstate-disconnected" id="id-a-purchaseclient.connectionstate-disconnected"></a>

```
int DISCONNECTED
```

서비스와의 연결이 끊어진 상태입니다.

Constant Value: 0


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/annotations/purchaseclient.featuretype.md
# PurchaseClient.FeatureType

```
public static @interface PurchaseClient.FeatureType
```

```
com.gaa.sdk.iap.PurchaseClient.FeatureType
```

지원되는 기능 유형

## Constants <a href="#id-a-purchaseclient.featuretype-constants" id="id-a-purchaseclient.featuretype-constants"></a>

***

### SUBSCRIPTIONS <a href="#id-a-purchaseclient.featuretype-subscriptions" id="id-a-purchaseclient.featuretype-subscriptions"></a>

```
String SUBSCRIPTIONS
```

구독 상품을 구입/관리 할 수 있습니다.

Constant Value: "subscriptions"


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/annotations/purchaseclient.producttype.md
# PurchaseClient.ProductType

```
public static @interface PurchaseClient.ProductType
```

```
com.gaa.sdk.iap.PurchaseClient.ProductType
```

지원되는 상품유형

## Constants <a href="#id-a-purchaseclient.producttype-constants" id="id-a-purchaseclient.producttype-constants"></a>

***

### ALL <a href="#id-a-purchaseclient.producttype-all" id="id-a-purchaseclient.producttype-all"></a>

```
String ALL
```

상품상세 정보를 한번에 받기 위한 타입

Constant Value: "all"

### INAPP <a href="#id-a-purchaseclient.producttype-inapp" id="id-a-purchaseclient.producttype-inapp"></a>

```
String INAPP
```

관리형 상품 타입

Constant Value: "inapp"

### AUTO <a href="#id-a-purchaseclient.producttype-auto" id="id-a-purchaseclient.producttype-auto"></a>

```
String AUTO
```

월정액 상품 타입

Constant Value: "auto"

### SUBS <a href="#id-a-purchaseclient.producttype-subs" id="id-a-purchaseclient.producttype-subs"></a>

```
String SUBS
```

구독형 상품 타입

Constant Value: "subscription"


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/annotations/purchaseclient.recurringaction.md
# PurchaseClient.RecurringAction

```
public static @interface PurchaseClient.RecurringAction
```

```
com.gaa.sdk.iap.PurchaseClient.RecurringAction
```

월정액 상품의 상태를 관리하기 위한 타입

## Constants <a href="#id-a-purchaseclient.recurringaction-constants" id="id-a-purchaseclient.recurringaction-constants"></a>

***

### CANCEL <a href="#id-a-purchaseclient.recurringaction-cancel" id="id-a-purchaseclient.recurringaction-cancel"></a>

```
String CANCEL
```

월정액 상품 자동결제 취소

Constant Value: "cancel"

### REACTIVATE <a href="#id-a-purchaseclient.recurringaction-reactivate" id="id-a-purchaseclient.recurringaction-reactivate"></a>

```
String REACTIVATE
```

월정액 상품 자동결제 취소 해제

Constant Value: "reactivate"


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/annotations/purchaseclient.responsecode.md
# PurchaseClient.ResponseCode

```
public static @interface PurchaseClient.ResponseCode
```

```
com.gaa.sdk.iap.PurchaseClient.ResponseCode
```

SDK에 사용되는 응답코드

## Constants <a href="#id-a-purchaseclient.responsecode-constants" id="id-a-purchaseclient.responsecode-constants"></a>

***

### RESULT\_OK <a href="#id-a-purchaseclient.responsecode-result_ok" id="id-a-purchaseclient.responsecode-result_ok"></a>

```
int RESULT_OK
```

성공

Constant Value: 0

### USER\_CANCELED <a href="#id-a-purchaseclient.responsecode-user_canceled" id="id-a-purchaseclient.responsecode-user_canceled"></a>

```
int USER_CANCELED
```

결제가 취소되었습니다.

Constant Value: 1

### RESULT\_SERVICE\_UNAVAILABLE <a href="#id-a-purchaseclient.responsecode-result_service_unavailable" id="id-a-purchaseclient.responsecode-result_service_unavailable"></a>

```
int RESULT_SERVICE_UNAVAILABLE
```

단말 또는 서버 네트워크 오류가 발생하였습니다.

Constant Value: 2

### RESULT\_BILLING\_UNAVAILABLE <a href="#id-a-purchaseclient.responsecode-result_billing_unavailable" id="id-a-purchaseclient.responsecode-result_billing_unavailable"></a>

```
int RESULT_BILLING_UNAVAILABLE
```

구매처리 과정에서 오류가 발생하였습니다.

Constant Value: 3

### RESULT\_ITEM\_UNAVAILABLE <a href="#id-a-purchaseclient.responsecode-result_item_unavailable" id="id-a-purchaseclient.responsecode-result_item_unavailable"></a>

```
int RESULT_ITEM_UNAVAILABLE
```

상품이 판매중이 아니거나 구매할 수 없는 상품입니다.

Constant Value: 4

### RESULT\_DEVELOPER\_ERROR <a href="#id-a-purchaseclient.responsecode-result_developer_error" id="id-a-purchaseclient.responsecode-result_developer_error"></a>

```
int RESULT_DEVELOPER_ERROR
```

올바르지 않은 요청입니다.

Constant Value: 5

### RESULT\_ERROR <a href="#id-a-purchaseclient.responsecode-result_error" id="id-a-purchaseclient.responsecode-result_error"></a>

```
int RESULT_ERROR
```

정의되지 않은 기타 오류가 발생하였습니다.

Constant Value: 6

### RESULT\_ITEM\_ALREADY\_OWNED <a href="#id-a-purchaseclient.responsecode-result_item_already_owned" id="id-a-purchaseclient.responsecode-result_item_already_owned"></a>

```
int RESULT_ITEM_ALREADY_OWNED
```

이미 아이템을 소유하고 있습니다.

Constant Value: 7

### RESULT\_ITEM\_NOT\_OWNED <a href="#id-a-purchaseclient.responsecode-result_item_not_owned" id="id-a-purchaseclient.responsecode-result_item_not_owned"></a>

```
int RESULT_ITEM_NOT_OWNED
```

아이템을 소유하고 있지 않아 소비할 수 없습니다.

Constant Value: 8

### RESULT\_FAIL <a href="#id-a-purchaseclient.responsecode-result_fail" id="id-a-purchaseclient.responsecode-result_fail"></a>

```
int RESULT_FAIL
```

결제에 실패했습니다. 결제 가능여부 및 결제수단을 확인 후 다시 결제해주세요.

Constant Value: 9

### RESULT\_NEED\_LOGIN <a href="#id-a-purchaseclient.responsecode-result_need_login" id="id-a-purchaseclient.responsecode-result_need_login"></a>

```
int RESULT_NEED_LOGIN
```

스토어 앱 로그인이 필요합니다.

Constant Value: 10

### RESULT\_NEED\_UPDATE <a href="#id-a-purchaseclient.responsecode-result_need_update" id="id-a-purchaseclient.responsecode-result_need_update"></a>

```
int RESULT_NEED_UPDATE
```

결제 모듈의 업데이트가 필요합니다.

Constant Value: 11

### RESULT\_SECURITY\_ERROR <a href="#id-a-purchaseclient.responsecode-result_security_error" id="id-a-purchaseclient.responsecode-result_security_error"></a>

```
int RESULT_SECURITY_ERROR
```

비정상 앱에서 결제가 요청되었습니다.

Constant Value: 12

### RESULT\_BLOCKED\_APP <a href="#id-a-purchaseclient.responsecode-result_blocked_app" id="id-a-purchaseclient.responsecode-result_blocked_app"></a>

```
int RESULT_BLOCKED_APP
```

요청이 차단되었습니다.

Constant Value: 13

### RESULT\_NOT\_SUPPORT\_SANDBOX <a href="#id-a-purchaseclient.responsecode-result_not_support_sandbox" id="id-a-purchaseclient.responsecode-result_not_support_sandbox"></a>

```
int RESULT_NOT_SUPPORT_SANDBOX
```

테스트 환경에서는 지원하지 않는 기능입니다.

Constant Value: 14

### ERROR\_DATA\_PARSING <a href="#id-a-purchaseclient.responsecode-error_data_parsing" id="id-a-purchaseclient.responsecode-error_data_parsing"></a>

```
int ERROR_DATA_PARSING
```

응답 데이터 분석 오류가 발생했습니다.

Constant Value: 1001

### ERROR\_SIGNATURE\_VERIFICATION <a href="#id-a-purchaseclient.responsecode-error_signature_verification" id="id-a-purchaseclient.responsecode-error_signature_verification"></a>

```
int ERROR_SIGNATURE_VERIFICATION
```

구매정보의 서명검증 에러가 발생했습니다.

Constant Value: 1002

### ERROR\_ILLEGAL\_ARGUMENT <a href="#id-a-purchaseclient.responsecode-error_illegal_argument" id="id-a-purchaseclient.responsecode-error_illegal_argument"></a>

```
int ERROR_ILLEGAL_ARGUMENT
```

정상적이지 않은 파라미터가 입력 되었습니다.

Constant Value: 1003

### ERROR\_UNDEFINED\_CODE <a href="#id-a-purchaseclient.responsecode-error_undefined_code" id="id-a-purchaseclient.responsecode-error_undefined_code"></a>

```
int ERROR_UNDEFINED_CODE
```

정의되지 않은 오류가 발생했습니다.

Constant Value: 1004

### ERROR\_SIGNATURE\_NOT\_VALIDATION <a href="#id-a-purchaseclient.responsecode-error_signature_not_validation" id="id-a-purchaseclient.responsecode-error_signature_not_validation"></a>

```
int ERROR_SIGNATURE_NOT_VALIDATION
```

입력한 라이센스키가 유효하지 않습니다.

Constant Value: 1005

### ERROR\_UPDATE\_OR\_INSTALL <a href="#id-a-purchaseclient.responsecode-error_update_or_install" id="id-a-purchaseclient.responsecode-error_update_or_install"></a>

```
int ERROR_UPDATE_OR_INSTALL
```

결제 모듈 설치에 실패하였습니다.

Constant Value: 1006

### ERROR\_SERVICE\_DISCONNECTED <a href="#id-a-purchaseclient.responsecode-error_service_disconnected" id="id-a-purchaseclient.responsecode-error_service_disconnected"></a>

```
int ERROR_SERVICE_DISCONNECTED
```

결제 모듈과의 연결이 끊어졌습니다.

Constant Value: 1007

### ERROR\_FEATURE\_NOT\_SUPPORTED <a href="#id-a-purchaseclient.responsecode-error_feature_not_supported" id="id-a-purchaseclient.responsecode-error_feature_not_supported"></a>

```
int ERROR_FEATURE_NOT_SUPPORTED
```

기능을 지원하지 않습니다.

Constant Value: 1008

### ERROR\_SERVICE\_TIMEOUT <a href="#id-a-purchaseclient.responsecode-error_service_timeout" id="id-a-purchaseclient.responsecode-error_service_timeout"></a>

```
int ERROR_SERVICE_TIMEOUT
```

서비스와 통신하는 시간이 초과되었습니다.

Constant Value: 1009

### ERROR\_CLIENT\_NOT\_ENABLED <a href="#id-a-purchaseclient.responsecode-error_client_not_enabled" id="id-a-purchaseclient.responsecode-error_client_not_enabled"></a>

```
int ERROR_CLIENT_NOT_ENABLED
```

서비스앱과의 바인딩을 할 수 없는 상태입니다.

Constant Value: 1010

### RESULT\_EMERGENY\_ERROR <a href="#id-a-purchaseclient.responsecode-result_emergeny_error" id="id-a-purchaseclient.responsecode-result_emergeny_error"></a>

```
int RESULT_EMERGENY_ERROR
```

서버 점검중입니다.

Constant Value: 99999


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/annotations/purchasedata.acknowledgestate.md
# PurchaseData.AcknowledgeState

```
public static @interface PurchaseData.AcknowledgeState
```

```
com.gaa.sdk.iap.PurchaseData.AcknowledgeState
```

구매확인에 대한 상태값

## Constants <a href="#id-a-purchasedata.acknowledgestate-constants" id="id-a-purchasedata.acknowledgestate-constants"></a>

***

### NOT\_ACKNOWLEDGED <a href="#id-a-purchasedata.acknowledgestate-not_acknowledged" id="id-a-purchasedata.acknowledgestate-not_acknowledged"></a>

```
int NOT_ACKNOWLEDGED
```

구매확인 되지 않음

Constant Value: 0

### ACKNOWLEDGED <a href="#id-a-purchasedata.acknowledgestate-acknowledged" id="id-a-purchasedata.acknowledgestate-acknowledged"></a>

```
int ACKNOWLEDGED
```

구매확인 됨

Constant Value: 1


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/annotations/purchasedata.purchasestate.md
# PurchaseData.PurchaseState

```
public static @interface PurchaseData.PurchaseState
```

```
com.gaa.sdk.iap.PurchaseData.PurchaseState
```

구매한 데이터의 상태값

## Constants <a href="#id-a-purchasedata.purchasestate-constants" id="id-a-purchasedata.purchasestate-constants"></a>

***

### PURCHASED <a href="#id-a-purchasedata.purchasestate-purchased" id="id-a-purchasedata.purchasestate-purchased"></a>

```
int PURCHASED
```

구매됨

Constant Value: 0

### CANCEL <a href="#id-a-purchasedata.purchasestate-cancel" id="id-a-purchasedata.purchasestate-cancel"></a>

```
int CANCEL
```

취소

Constant Value: 1

### REFUND <a href="#id-a-purchasedata.purchasestate-refund" id="id-a-purchasedata.purchasestate-refund"></a>

```
int REFUND
```

환불

Constant Value: 2


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/annotations/purchasedata.recurringstate.md
# PurchaseData.RecurringState

```
public static @interface PurchaseData.RecurringState
```

```
com.gaa.sdk.iap.PurchaseData.RecurringState
```

월정액 상품의 상태값

## Constants <a href="#id-a-purchasedata.recurringstate-constants" id="id-a-purchasedata.recurringstate-constants"></a>

***

### RECURRING <a href="#id-a-purchasedata.recurringstate-recurring" id="id-a-purchasedata.recurringstate-recurring"></a>

```
int RECURRING
```

자동 결제 중

Constant Value: 0

### CANCEL <a href="#id-a-purchasedata.recurringstate-cancel" id="id-a-purchasedata.recurringstate-cancel"></a>

```
int CANCEL
```

자동 결제 해지 예약중

Constant Value: 1

### NON\_AUTO\_PRODUCT <a href="#id-a-purchasedata.recurringstate-non_auto_product" id="id-a-purchasedata.recurringstate-non_auto_product"></a>

```
int NON_AUTO_PRODUCT
```

자동결제가 아닌 경우

Constant Value: -1


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/annotations/purchaseflowparams.prorationmode.md
# PurchaseFlowParams.ProrationMode

```
public static @interface PurchaseFlowParams.ProrationMode
```

```
com.gaa.sdk.iap.PurchaseFlowParams.ProrationMode
```

정기 결제의 업그레이드 또는 다운그레이드를 위한 비례 배분 모드 입니다.

## Constants <a href="#id-a-purchaseflowparams.prorationmode-constants" id="id-a-purchaseflowparams.prorationmode-constants"></a>

***

### UNKNOWN\_SUBSCRIPTION\_UPGRADE\_DOWNGRADE\_POLICY <a href="#id-a-purchaseflowparams.prorationmode-unknown_subscription_upgrade_downgrade_policy" id="id-a-purchaseflowparams.prorationmode-unknown_subscription_upgrade_downgrade_policy"></a>

```
int UNKNOWN_SUBSCRIPTION_UPGRADE_DOWNGRADE_POLICY
```

명시되지 않은 상태값

Constant Value: 0

### IMMEDIATE\_WITH\_TIME\_PRORATION <a href="#id-a-purchaseflowparams.prorationmode-immediate_with_time_proration" id="id-a-purchaseflowparams.prorationmode-immediate_with_time_proration"></a>

```
int IMMEDIATE_WITH_TIME_PRORATION
```

교체가 즉시 적용되며, 새로운 만료 시간은 비례 배분되어 사용자에게 입금되거나 청구됩니다. (default)

Constant Value: 1

### IMMEDIATE\_AND\_CHARGE\_PRORATED\_PRICE <a href="#id-a-purchaseflowparams.prorationmode-immediate_and_charge_prorated_price" id="id-a-purchaseflowparams.prorationmode-immediate_and_charge_prorated_price"></a>

```
int IMMEDIATE_AND_CHARGE_PRORATED_PRICE
```

교체가 즉시 적용되며 청구 주기는 동일하게 유지됩니다. 나머지 기간에 대한 가격이 청구됩니다.

Constant Value: 2

### IMMEDIATE\_WITHOUT\_PRORATION <a href="#id-a-purchaseflowparams.prorationmode-immediate_without_proration" id="id-a-purchaseflowparams.prorationmode-immediate_without_proration"></a>

```
int IMMEDIATE_WITHOUT_PRORATION
```

교체가 즉시 적용되며 다음 결제일에 새로운 가격이 청구됩니다. 청구 주기는 동일하게 유지됩니다.

Constant Value: 3

### DEFERRED <a href="#id-a-purchaseflowparams.prorationmode-deferred" id="id-a-purchaseflowparams.prorationmode-deferred"></a>

```
int DEFERRED
```

기존 요금제가 만료되면 교체가 적용되며 새 요금이 동시에 청구됩니다.

Constant Value: 4


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes.md
# Classes

- [AcknowledgeParams](/dev/tools/billing/v21/references/classes/acknowledgeparams.md)
- [AcknowledgeParams.Builder](/dev/tools/billing/v21/references/classes/acknowledgeparams.builder.md)
- [ConsumeParams](/dev/tools/billing/v21/references/classes/consumeparams.md)
- [ConsumeParams.Builder](/dev/tools/billing/v21/references/classes/consumeparams.builder.md)
- [IapResult](/dev/tools/billing/v21/references/classes/iapresult.md)
- [IapResult.Builder](/dev/tools/billing/v21/references/classes/iapresult.builder.md)
- [ProductDetail](/dev/tools/billing/v21/references/classes/productdetail.md)
- [ProductDetailsParams](/dev/tools/billing/v21/references/classes/productdetailsparams.md)
- [ProductDetailsParams.Builder](/dev/tools/billing/v21/references/classes/productdetailsparams.builder.md)
- [PurchaseClient](/dev/tools/billing/v21/references/classes/purchaseclient.md)
- [PurchaseClient.Builder](/dev/tools/billing/v21/references/classes/purchaseclient.builder.md)
- [PurchaseData](/dev/tools/billing/v21/references/classes/purchasedata.md)
- [PurchaseFlowParams](/dev/tools/billing/v21/references/classes/purchaseflowparams.md)
- [PurchaseFlowParams.Builder](/dev/tools/billing/v21/references/classes/purchaseflowparams.builder.md)
- [RecurringProductParams](/dev/tools/billing/v21/references/classes/recurringproductparams.md)
- [RecurringProductParams.Builder](/dev/tools/billing/v21/references/classes/recurringproductparams.builder.md)
- [SubscriptionParams](/dev/tools/billing/v21/references/classes/subscriptionparams.md)
- [SubscriptionParams.Builder](/dev/tools/billing/v21/references/classes/subscriptionparams.builder.md)
- [SubscriptionUpdateParams](/dev/tools/billing/v21/references/classes/subscriptionupdateparams.md)
- [SubscriptionUpdateParams.Builder](/dev/tools/billing/v21/references/classes/subscriptionupdateparams.builder.md)


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/acknowledgeparams.md
# AcknowledgeParams

```
public class AcknowledgeParams extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.AcknowledgeParams
```

구매한 인앱 상품을 확인하기 위한 파라미터 입니다.

**See Also:**

PurchaseClient.acknowledgeAsync(AcknowledgeParams, \[I]AcknowledgeListener)

## Summary <a href="#id-c-acknowledgeparams-summary" id="id-c-acknowledgeparams-summary"></a>

***

| Nested classes |                                                                                                                             |
| -------------- | --------------------------------------------------------------------------------------------------------------------------- |
| class          | <p><a href="acknowledgeparams.builder">[C]AcknowledgeParams.Builder</a></p><p>AcknowledgeParams의 인스턴스를 쉽게 만들기 위한 빌더입니다.</p> |

## Public methods <a href="#id-c-acknowledgeparams-publicmethods" id="id-c-acknowledgeparams-publicmethods"></a>

***

### getPurchaseData <a href="#id-c-acknowledgeparams-getpurchasedata" id="id-c-acknowledgeparams-getpurchasedata"></a>

```
[C]PurchaseData getPurchaseData()
```

인앱 상품의 구매 데이터를 반환합니다.

| **Returns:**                      |             |
| --------------------------------- | ----------- |
| [\[C\]PurchaseData](purchasedata) | <p><br></p> |

### getDeveloperPayload <a href="#id-c-acknowledgeparams-getdeveloperpayload" id="id-c-acknowledgeparams-getdeveloperpayload"></a>

```
String getDeveloperPayload()
```

개발사에서 지정한 페이로드를 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### newBuilder <a href="#id-c-acknowledgeparams-newbuilder" id="id-c-acknowledgeparams-newbuilder"></a>

```
[C]AcknowledgeParams.Builder newBuilder()
```

AcknowledgeParams의 인스턴스를 만들기 위한 빌더를 생성합니다.

| **Returns:**                                                |             |
| ----------------------------------------------------------- | ----------- |
| [\[C\]AcknowledgeParams.Builder](acknowledgeparams.builder) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/acknowledgeparams.builder.md
# AcknowledgeParams.Builder

```
public final class AcknowledgeParams.Builder extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.AcknowledgeParams.Builder
```

[\[C\]AcknowledgeParams](acknowledgeparams)의 인스턴스를 쉽게 만들기 위한 빌더입니다.

## Public methods <a href="#id-c-acknowledgeparams.builder-publicmethods" id="id-c-acknowledgeparams.builder-publicmethods"></a>

***

### setPurchaseData <a href="#id-c-acknowledgeparams.builder-setpurchasedata" id="id-c-acknowledgeparams.builder-setpurchasedata"></a>

```
AcknowledgeParams.Builder setPurchaseData([C]PurchaseData purchaseData)
```

| **Parameters:**           |             |
| ------------------------- | ----------- |
| purchaseData              | <p><br></p> |
| **Returns:**              |             |
| AcknowledgeParams.Builder | <p><br></p> |

### setDeveloperPayload <a href="#id-c-acknowledgeparams.builder-setdeveloperpayload" id="id-c-acknowledgeparams.builder-setdeveloperpayload"></a>

```
AcknowledgeParams.Builder setDeveloperPayload(String developerPayload)
```

| **Parameters:**           |                |
| ------------------------- | -------------- |
| developerPayload          | 개발사의 페이로드 입니다. |
| **Returns:**              |                |
| AcknowledgeParams.Builder | <p><br></p>    |

### build <a href="#id-c-acknowledgeparams.builder-build" id="id-c-acknowledgeparams.builder-build"></a>

[\[C\]AcknowledgeParams](acknowledgeparams) build()

AcknowledgeParams의 인스턴스를 생성합니다.

| **Returns:**                                 |             |
| -------------------------------------------- | ----------- |
| [\[C\]AcknowledgeParams](acknowledgeparams)  | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/consumeparams.md
# ConsumeParams

```
public class ConsumeParams extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.ConsumeParams
```

구매한 인앱 상품을 소비하기 위한 파라미터 입니다.

**See Also:**

[PurchaseClient.consumeAsync](purchaseclient)(ConsumeParams, [\[I\]ConsumeListener](../interfaces/consumelistener))\


## Summary <a href="#id-c-consumeparams-summary" id="id-c-consumeparams-summary"></a>

***

| Nested classes |                                                                             |
| -------------- | --------------------------------------------------------------------------- |
| class          | <p>[C]ConsumeParams.Builder</p><p>ConsumeParams의 인스턴스를 쉽게 만들기 위한 빌더입니다.</p> |

## Public methods <a href="#id-c-consumeparams-publicmethods" id="id-c-consumeparams-publicmethods"></a>

***

### getPurchaseData <a href="#id-c-consumeparams-getpurchasedata" id="id-c-consumeparams-getpurchasedata"></a>

```
[C]PurchaseData getPurchaseData()
```

인앱 상품의 구매 데이터를 반환합니다.

| **Returns:**                      |             |
| --------------------------------- | ----------- |
| [\[C\]PurchaseData](purchasedata) | <p><br></p> |

### getDeveloperPayload <a href="#id-c-consumeparams-getdeveloperpayload" id="id-c-consumeparams-getdeveloperpayload"></a>

```
String getDeveloperPayload()
```

개발사에서 지정한 페이로드를 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### newBuilder <a href="#id-c-consumeparams-newbuilder" id="id-c-consumeparams-newbuilder"></a>

```
[C]ConsumeParams.Builder newBuilder()
```

ConsumeParams의 인스턴스를 만들기 위한 빌더를 생성합니다.

| **Returns:**                                        |             |
| --------------------------------------------------- | ----------- |
| [\[C\]ConsumeParams.Builder](consumeparams.builder) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/consumeparams.builder.md
# ConsumeParams.Builder

```
public final class ConsumeParams.Builder extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.ConsumeParams.Builder
```

[\[C\]ConsumeParams](consumeparams)의 인스턴스를 쉽게 만들기 위한 빌더입니다.

## Public methods <a href="#id-c-consumeparams.builder-publicmethods" id="id-c-consumeparams.builder-publicmethods"></a>

***

### setPurchaseData <a href="#id-c-consumeparams.builder-setpurchasedata" id="id-c-consumeparams.builder-setpurchasedata"></a>

```
ConsumeParams.Builder setPurchaseData([C]PurchaseData purchaseData)
```

| **Parameters:**       |             |
| --------------------- | ----------- |
| purchaseData          | <p><br></p> |
| **Returns:**          |             |
| ConsumeParams.Builder | <p><br></p> |

### setDeveloperPayload <a href="#id-c-consumeparams.builder-setdeveloperpayload" id="id-c-consumeparams.builder-setdeveloperpayload"></a>

```
ConsumeParams.Builder setDeveloperPayload(String developerPayload)
```

| **Parameters:**       |                |
| --------------------- | -------------- |
| developerPayload      | 개발사의 페이로드 입니다. |
| **Returns:**          |                |
| ConsumeParams.Builder | <p><br></p>    |

### build <a href="#id-c-consumeparams.builder-build" id="id-c-consumeparams.builder-build"></a>

```
[C]ConsumeParams build()
```

ConsumeParams의 인스턴스를 생성합니다.

| **Returns:**                        |             |
| ----------------------------------- | ----------- |
| [\[C\]ConsumeParams](consumeparams) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/iapresult.md
# IapResult

```
public final class IapResult extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.IapResult
```

In-app Purchase API 응답의 코드 및 메시지가 포함된 객체 입니다.

## Summary <a href="#id-c-iapresult-summary" id="id-c-iapresult-summary"></a>

***

| **Nested classes** |                                                                                                     |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| class              | <p><a href="iapresult.builder">[C]IapResult.Builder</a></p><p>IapResult의 인스턴스를 쉽게 만들기 위한 빌더입니다.</p> |

## Public methods <a href="#id-c-iapresult-publicmethods" id="id-c-iapresult-publicmethods"></a>

***

### getResponseCode <a href="#id-c-iapresult-getresponsecode" id="id-c-iapresult-getresponsecode"></a>

```
int getResponseCode()
```

In-app Purchase API 응답 코드를 반환합니다.

| **Returns:** |                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------- |
| int          | [\[A\]PurchaseClient.ResponseCode](../annotations/purchaseclient.responsecode)에 해당되는 결과 값 |

### getMessage <a href="#id-c-iapresult-getmessage" id="id-c-iapresult-getmessage"></a>

```
String getMessage()
```

In-app Purchase API 응답 메시지를 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### isSuccess <a href="#id-c-iapresult-issuccess" id="id-c-iapresult-issuccess"></a>

```
boolean isSuccess()
```

In-app Purchase API 응답 코드가 성공(PurchaseClient.ResponseCode.RESULT\_OK) 인지를 나타냅니다.

| **Returns:** |             |
| ------------ | ----------- |
| boolean      | <p><br></p> |

### isFailure <a href="#id-c-iapresult-isfailure" id="id-c-iapresult-isfailure"></a>

```
boolean isFailure()
```

In-app Purchase API 응답 코드가 실패 인지를 나타냅니다.

| **Returns:** |             |
| ------------ | ----------- |
| boolean      | <p><br></p> |

### newBuilder <a href="#id-c-iapresult-newbuilder" id="id-c-iapresult-newbuilder"></a>

```
[C]IapResult.Builder newBuilder()
```

IapResult의 인스턴스를 만들기 위한 빌더를 생성합니다.

| **Returns:**                                |             |
| ------------------------------------------- | ----------- |
| [\[C\]IapResult.Builder](iapresult.builder) | <p><br></p> |

\


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/iapresult.builder.md
# IapResult.Builder

```
public final class IapResult.Builder extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.IapResult.Builder
```

[`IapResult`](iapresult)의 인스턴스를 쉽게 만들기 위한 빌더입니다.

## Public methods <a href="#id-c-iapresult.builder-publicmethods" id="id-c-iapresult.builder-publicmethods"></a>

***

### setResponseCode <a href="#id-c-iapresult.builder-setresponsecode" id="id-c-iapresult.builder-setresponsecode"></a>

```
IapResult.Builder setResponseCode(int responseCode)
```

| **Parameters:**   |                               |
| ----------------- | ----------------------------- |
| responseCode      | In-app Purchase API 응답 코드입니다. |
| **Returns:**      |                               |
| IapResult.Builder | <p><br></p>                   |

### setMessage <a href="#id-c-iapresult.builder-setmessage" id="id-c-iapresult.builder-setmessage"></a>

```
IapResult.Builder setMessage(String message)
```

<table data-header-hidden><thead><tr><th></th><th></th></tr></thead><tbody><tr><td><strong>Parameters:</strong></td><td></td></tr><tr><td><pre><code>message
</code></pre></td><td>In-app Purchase API 응답 메시지입니다.</td></tr><tr><td><strong>Returns:</strong></td><td></td></tr><tr><td>IapResult.Builder</td><td><br></td></tr></tbody></table>

### build <a href="#id-c-iapresult.builder-build" id="id-c-iapresult.builder-build"></a>

```
[C]IapResult build()
```

IapResult의 인스턴스를 생성합니다.\


| **Returns:**                |             |
| --------------------------- | ----------- |
| [\[C\]IapResult](iapresult) | <p><br></p> |



출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/productdetail.md
# ProductDetail

```
public class ProductDetail extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.ProductDetail
```

인앱결제에 대한 구매 데이터를 나타냅니다.

## Summary <a href="#id-c-productdetail-summary" id="id-c-productdetail-summary"></a>

***

<table><thead><tr><th>Public constructors</th></tr></thead><tbody><tr><td><pre><code>ProductDetail(String jsonProductDetail)
</code></pre></td></tr></tbody></table>

## Public methods <a href="#id-c-productdetail-publicmethods" id="id-c-productdetail-publicmethods"></a>

***

### getOriginalJson <a href="#id-c-productdetail-getoriginaljson" id="id-c-productdetail-getoriginaljson"></a>

```
String getOriginalJson()
```

인앱 상품의 세부 정보가 포함된 JSON 형식의 문자열을 반환 합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getProductId <a href="#id-c-productdetail-getproductid" id="id-c-productdetail-getproductid"></a>

```
String getProductId()
```

구매한 인앱상품의 아이디를 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getType <a href="#id-c-productdetail-gettype" id="id-c-productdetail-gettype"></a>

```
String getType()
```

인앱 상품의 타입을 반환합니다.

| **Returns:** |                                                                              |
| ------------ | ---------------------------------------------------------------------------- |
| String       | [\[A\]PurchaseClient.ProductType](../annotations/purchaseclient.producttype) |

### getPrice <a href="#id-c-productdetail-getprice" id="id-c-productdetail-getprice"></a>

```
String getPrice()
```

인앱 상품의 가격을 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getPriceCurrencyCode <a href="#id-c-productdetail-getpricecurrencycode" id="id-c-productdetail-getpricecurrencycode"></a>

```
String getPriceCurrencyCode()
```

가격에 대한 ISO 4217 통화 코드를 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getPriceAmountMicros <a href="#id-c-productdetail-getpriceamountmicros" id="id-c-productdetail-getpriceamountmicros"></a>

```
String getPriceAmountMicros()
```

가격을 마이크로 단위로 반환합니다. 1,000,000 마이크로 단위는 통화의 한 단위와 같습니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getTitle <a href="#id-c-productdetail-gettitle" id="id-c-productdetail-gettitle"></a>

```
String getTitle()
```

인앱 상품의 이름을 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/productdetailsparams.md
# ProductDetailsParams

```
public class ProductDetailsParams extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.ProductDetailsParams
```

인앱 상품 상세정보를 요청하기 위한 파라미터 입니다.

**See Also:**

[PurchaseClient.queryProductDetailsAsync](purchaseclient)(ProductDetailsParams, [\[I\]ProductDetailsListener](../interfaces/productdetailslistener))

## Summary <a href="#id-c-productdetailsparams-summary" id="id-c-productdetailsparams-summary"></a>

***

| Nested classes |                                                                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| class          | <p><a href="../../../old-version/v19/undefined-3/classes/productdetailsparams.builder">ProductDetailsParams.Builder</a></p><p>ProductDetailsParams의 인스턴스를 쉽게 만들기 위한 빌더입니다.</p> |

## Public methods <a href="#id-c-productdetailsparams-publicmethods" id="id-c-productdetailsparams-publicmethods"></a>

***

### getProductIdList <a href="#id-c-productdetailsparams-getproductidlist" id="id-c-productdetailsparams-getproductidlist"></a>

```
List<String> getProductIdList()
```

인앱 상품의 아이디 리스트를 반환합니다.\


| **Returns:**  |             |
| ------------- | ----------- |
| List\<String> | <p><br></p> |

### getProductType <a href="#id-c-productdetailsparams-getproducttype" id="id-c-productdetailsparams-getproducttype"></a>

```
String getProductType()
```

인앱 상품의 아이디 리스트들의 타입을 반환합니다.

| **Returns:** |                                                                              |
| ------------ | ---------------------------------------------------------------------------- |
| String       | [\[A\]PurchaseClient.ProductType](../annotations/purchaseclient.producttype) |

### newBuilder <a href="#id-c-productdetailsparams-newbuilder" id="id-c-productdetailsparams-newbuilder"></a>

```
ProductDetailsParams.Builder newBuilder()
```

ProductDetailsParams의 인스턴스를 만들기 위한 빌더를 생성합니다.

| **Returns:**                                                                                              |             |
| --------------------------------------------------------------------------------------------------------- | ----------- |
| [ProductDetailsParams.Builder](../../../old-version/v19/undefined-3/classes/productdetailsparams.builder) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/productdetailsparams.builder.md
# ProductDetailsParams.Builder

```
public final class ProductDetailsParams.Builder extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.ProductDetailsParams.Builder
```

[ProductDetailsParams](productdetailsparams)의 인스턴스를 쉽게 만들기 위한 빌더입니다.

## Public methods <a href="#id-c-productdetailsparams.builder-publicmethods" id="id-c-productdetailsparams.builder-publicmethods"></a>

***

### setProductIdList <a href="#id-c-productdetailsparams.builder-setproductidlist" id="id-c-productdetailsparams.builder-setproductidlist"></a>

```
ProductDetailsParams.Builder setProductIdList(List<String> productIdList)
```

<table data-header-hidden><thead><tr><th></th><th></th></tr></thead><tbody><tr><td><strong>Parameters:</strong></td><td></td></tr><tr><td><pre><code>productIdList
</code></pre></td><td>인앱 상품 아이디 리스트</td></tr><tr><td><strong>Returns:</strong></td><td></td></tr><tr><td>ProductDetailsParams.Builder</td><td><br></td></tr></tbody></table>

### setProductType <a href="#id-c-productdetailsparams.builder-setproducttype" id="id-c-productdetailsparams.builder-setproducttype"></a>

```
ProductDetailsParams.Builder setProductType(String productType)
```

<table data-header-hidden><thead><tr><th></th><th></th></tr></thead><tbody><tr><td><strong>Parameters:</strong></td><td></td></tr><tr><td><pre><code>productType
</code></pre></td><td><p>인앱 상품들의 상품 타입</p><p><a href="../annotations/purchaseclient.producttype">[A]PurchaseClient.ProductType</a></p></td></tr><tr><td><strong>Returns:</strong></td><td></td></tr><tr><td>ProductDetailsParams.Builder</td><td><br></td></tr></tbody></table>

### build <a href="#id-c-productdetailsparams.builder-build" id="id-c-productdetailsparams.builder-build"></a>

```
 build()
```

ProductDetailsParams의 인스턴스를 생성합니다.

| **Returns:**                                 |             |
| -------------------------------------------- | ----------- |
| [ProductDetailsParams](productdetailsparams) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/purchaseclient.md
# PurchaseClient

```
public abstract class BillingClient extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.PurchaseClient
```

라이브러리와 사용자 어플리케이션 코드 간의 통신을 위한 기본 인터페이스 입니다.

모든 메서드는 비동기로 처리되며 UI thread에서 호출되어야 하며 모든 콜백은 UI thread로 반환됩니다.

이 클래스를 인스턴스화한 후 연결 설정을 하려면 `startConnection(`[`[I]PurchaseClientStateListener`](../interfaces/purchaseclientstatelistener)`)` 메서드를 호출하고 설정이 완료되면 리스너를 통해 알림을 받을 수 있습니다.

| `class` `YourPurchaseManager implements` `PurchaseClientStateListener {     private` `PurchaseClient mPurchaseClient;` `public` `YourPurchaseManager() {         mPurchaseClient = PurchaseClient.newBuilder(activity)                     .setBase64PublicKey(/* your public key */)                     .setListener(this)                     .build();     } ... }` |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

\
일반적으로 설정이 완료되면 구매내역 조회와 인앱 상품의 상세정보를 호출합니다. `queryPurchasesAsync(String,` [`QueryPurchasesListener`](../interfaces/querypurchaseslistener)`)` 및 `queryProductDetailsAsync(`[`ProductDetailsParams`](productdetailsparams)`,` [`ProductDetailsListener`](../interfaces/productdetailslistener)`)` 참조하세요.

이 객체로 모든 작업을 마치고 어플리케이션을 종료할 때는 `endConnection()`의 호출을 잊지마십시오. 결제 모듈과의 연결해제 및 자원을 해제해야 합니다. 권장 위치는 `Activity.onDestroy()` 입니다.

라이브러리의 자세한 로그를 확인하려면 `Logger.setLogLevel(int)` 에 `android.util.Log.VERBOSE` 입력하여 호출하면 됩니다.\
주의!!! 어플리케이션의 릴리즈 버전을 출시 전에 꼭! `Logger.setLogLevel(int)`를 제거 하십시오.

## Summary <a href="#id-c-purchaseclient-summary" id="id-c-purchaseclient-summary"></a>

***

| Nested classes |                                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------- |
| @interface     | [PurchaseClient.ConnectionState](../annotations/purchaseclient.connectionstate)                                |
| @interface     | [PurchaseClient.ResponseCode](../annotations/purchaseclient.responsecode)                                      |
| @interface     | [PurchaseClient.ProductType](../annotations/purchaseclient.producttype)                                        |
| @interface     | [PurchaseClient.RecurringAction](../annotations/purchaseclient.recurringaction)                                |
| @interface     | [PurchaseClient.FeatureType](../annotations/purchaseclient.featuretype)                                        |
| class          | <p><a href="purchaseclient.builder">PurchaseClient.Builder</a></p><p>PurchaseClient 인스턴스를 쉽게 만들기 위한 빌더입니다.</p> |

## Public methods <a href="#id-c-purchaseclient-publicmethods" id="id-c-purchaseclient-publicmethods"></a>

***

### newBuilder <a href="#id-c-purchaseclient-newbuilder" id="id-c-purchaseclient-newbuilder"></a>

```
@AnyThread
public static  newBuilder(Context context)
```

PurchaseClient의 인스턴스를 만들기 위한 빌더를 생성합니다.

| **Parameters:**                                  |                                              |
| ------------------------------------------------ | -------------------------------------------- |
| context                                          | applicationContext를 사용하여 결제모듈과 연결을 위해 사용됩니다. |
| Returns:                                         |                                              |
| [PurchaseClient.Builder](purchaseclient.builder) |                                              |

### isReady <a href="#id-c-purchaseclient-isready" id="id-c-purchaseclient-isready"></a>

```
@AnyThread
public abstract boolean isReady()
```

클라이언트가 현재 서비스에 연결되어 있는지 확인하여 다른 함수에 대한 요청이 성공하는지 확인합니다.

| Returns: |
| -------- |
| boolean  |

### getConnectionState <a href="#id-c-purchaseclient-getconnectionstate" id="id-c-purchaseclient-getconnectionstate"></a>

```
@AnyThread
public abstract int getConnectionState()
```

현재 PurchaseClient 의 연결상태를 반환합니다.

| Returns: |                                                                                              |
| -------- | -------------------------------------------------------------------------------------------- |
| int      | [PurchaseClient.ConnectionState](../annotations/purchaseclient.connectionstate) 에 해당하는 상태 값. |



### isFeatureSupported <a href="#id-c-purchaseclient-isfeaturesupported" id="id-c-purchaseclient-isfeaturesupported"></a>

```
@UiThread
public abstract [C]IapResult isFeatureSupported(String feature)
```

지정한  기능이 서비스에서 지원 되는지 확인합니다.

| **Parameters:** |                                                                                                                               |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| String          | [PurchaseClient.FeatureType](../annotations/purchaseclient.featuretype) 상수 중 하나 입니다.                                          |
| **Returns:**    |                                                                                                                               |
| IapResult       | <p>기능이 지원되는 경우 PurchaseClient.ResponseCode.RESULT_OK<br>그렇지 않을 경우 PurchaseClient.ResponseCode.ERROR_FEATURE_NOT_SUPPORTED</p> |

### startConnection <a href="#id-c-purchaseclient-startconnection" id="id-c-purchaseclient-startconnection"></a>

```
@AnyThread
public abstract void startConnection([I]PurchaseClientStateListener listener)
```

PurchaseClient 설정 프로세스를 비동기적으로 시작합니다.\
완료되면 [\[I\]PurchaseClientStateListener](../interfaces/purchaseclientstatelistener)를 통해 응답 받습니다.

| **Parameters:** |                                                |
| --------------- | ---------------------------------------------- |
| listener        | PurchaseClient 설정과 결제모듈 연결이 완료 되었을 때 응답을 받습니다. |

### endConnection <a href="#id-c-purchaseclient-endconnection" id="id-c-purchaseclient-endconnection"></a>

```
@AnyThread
public abstract void endConnection()
```

결제 모듈과의 연결을 끊고 PurchaseClient의 보유 자원을 해제 합니다.

### launchUpdateOrInstallFlow <a href="#id-c-purchaseclient-launchupdateorinstallflow" id="id-c-purchaseclient-launchupdateorinstallflow"></a>

```
@UiThread
public abstract void launchUpdateOrInstallFlow(Activity activity, [I]IapResultListener listener)
```

IapResult.getResponseCode()의 값이 PurchaseClient.ResponseCode.RESULT\_NEED\_UPDATE가 발생했을 때\
결제 모듈의 업데이트 또는 설치를 위해 호출해야 합니다.

| **Parameters:** |                                                           |
| --------------- | --------------------------------------------------------- |
| activity        | Activity: 결제 모듈의 업데이트 또는 설치의 흐름 알기위해 activity를 참조합니다.     |
| listener        | [\[I\]IapResultListener](../interfaces/iapresultlistener) |

### launchPurchaseFlow <a href="#id-c-purchaseclient-launchpurchaseflow" id="id-c-purchaseclient-launchpurchaseflow"></a>

```
@UiThread
public abstract [C]IapResult launchPurchaseFlow(Activity activity, [C]PurchaseFlowParams params)
```

인앱 상품 구매를 시작합니다.\
구매화면이 표시되고 결과를 PurchaseClient 를 초기화할 때 지정했던 [\[I\]PurchasesUpdatedListener](../interfaces/purchasesupdatedlistener)를 통해 응답 받습니다.

| **Parameters:**      |                                          |
| -------------------- | ---------------------------------------- |
| activity             | Activity: 구매화면을 띄우기 위해 activity를 참조합니다.  |
| params               | PurchaseFlowParams 구매화면을 띄우기 위한 매개변수입니다. |
| **Returns:**         |                                          |
| <p>IapResult<br></p> |                                          |

### consumeAsync <a href="#id-c-purchaseclient-consumeasync" id="id-c-purchaseclient-consumeasync"></a>

```
@AnyThread
public abstract void consumeAsync(params, listener)
```

구매한 상품을 소비 합니다.\
소유된 상품만 수행할 수 있으며, 소비를 진행했던 상품의 경우 다시 구매를 진행해야 합니다.\
경고! API v6 부터는 구매 후 3일 이내에 소비(consume) 또는 확인(acknowledge)을 하지 않으면 환불됩니다.

| **Parameters:** |                                                                      |
| --------------- | -------------------------------------------------------------------- |
| params          | [ConsumeParams](consumeparams) 소비를 위한 매개변수입니다.                       |
| listener        | [ConsumeListener](../interfaces/consumelistener) 소비 작업의 결과를 전달 받습니다. |

### acknowledgeAsync <a href="#id-c-purchaseclient-acknowledgeasync" id="id-c-purchaseclient-acknowledgeasync"></a>

```
@AnyThread
public abstract void acknowledgeAsync(acknowledgeParams, listener)
```

구매한 상품을 확인 합니다.\
소유된 상품만 수행할 수 있으며, 확인을 했던 상품은 소비를 하지 않으면 재구매를 할 수 없습니다.\
관리형 상품 또는 월정액 상품에 모두 적용가능 합니다.\
특히 관리형 상품의 경우 확인(acknowledge)를 하고 일정 기간후에 소비(consume)을 진행하는 기간제 상품으로 활용할 수 있습니다.\
경고! API v6 부터는 구매 후 3일 이내에 소비(consume) 또는 확인(acknowledge)을 하지 않으면 환불됩니다.

| **Parameters:**   |                                                                                 |
| ----------------- | ------------------------------------------------------------------------------- |
| acknowledgeParams | [AcknowledgeParams](acknowledgeparams) 구매 확인을 위한 매개변수입니다.                       |
| listener          | [AcknowledgeListener](../interfaces/acknowledgelistener) 구매 확인 작업의 결과를 전달 받습니다. |

### queryPurchasesAsync <a href="#id-c-purchaseclient-querypurchasesasync" id="id-c-purchaseclient-querypurchasesasync"></a>

```
@AnyThread
public abstract void queryPurchasesAsync(String productType,  listener)
```

앱에서 구매한 모든 상품에 대한 구매정보를 가져옵니다.

| **Parameters:** |                                                                                        |
| --------------- | -------------------------------------------------------------------------------------- |
| productType     | [PurchaseClient.ProductType](../annotations/purchaseclient.producttype) 상품의 타입을 지정합니다. |
| listener        | [QueryPurchasesListener](../interfaces/querypurchaseslistener) 구매내역에 대한 결과를 전달 받습니다.   |

### queryProductDetailsAsync <a href="#id-c-purchaseclient-queryproductdetailsasync" id="id-c-purchaseclient-queryproductdetailsasync"></a>

```
@AnyThread
public abstract void queryProductDetailsAsync(params, listener)
```

개발자 센터에 등록한 인앱상품의 상세정보를 가져옵니다.

| **Parameters:** |                                                                                         |
| --------------- | --------------------------------------------------------------------------------------- |
| params          | [ProductDetailsParams](productdetailsparams) 상품 상세정보를 가져오기 위한 매개변수입니다.                  |
| listener        | [ProductDetailsListener](../interfaces/productdetailslistener) 상품 상세정보에 대한 결과를 전달 받습니다. |

### manageRecurringProductAsync <a href="#id-c-purchaseclient-managerecurringproductasync" id="id-c-purchaseclient-managerecurringproductasync"></a>

```
@AnyThread
public abstract void manageRecurringProductAsync(recurringProductParams, listener)
```

월정액 상품의 상태를 변경 합니다.\
매월 자동결제를 취소 하거나 취소한 자동결제를 다시 재개하는 동작을 진행합니다.

| **Parameters:**        |                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| recurringProductParams | [RecurringProductParams](recurringproductparams) 월정액 상품 상태 변경을 위한 매개변수입니다.                       |
| listener               | [RecurringProductListener](../interfaces/recurringproductlistener) 월정액 상품 상태 변경에 대한 결과를 전달 받습니다. |

### launchLoginFlowAsync <a href="#id-c-purchaseclient-launchloginflowasync" id="id-c-purchaseclient-launchloginflowasync"></a>

```
@UiThread
public abstract void launchLoginFlowAsync(Activity activity, [I]IapResultListener listener)
```

IapResult.getResponseCode()의 값이 PurchaseClient.ResponseCode.RESULT\_NEED\_LOGIN가 발생했을 때 스토어 로그인을 진행합니다.

| **Parameters:** |                                                                                  |
| --------------- | -------------------------------------------------------------------------------- |
| activity        | Activity: 스토어 로그인의 흐름 알기위해 activity를 참조합니다.                                      |
| listener        | [IapResultListener](../interfaces/iapresultlistener) 로그인 성공, 실패에 대한 결과를 전달 받습니다. |

### launchManageSubscription <a href="#id-c-purchaseclient-launchmanagesubscription" id="id-c-purchaseclient-launchmanagesubscription"></a>

```
@UiThread
public abstract void launchManageSubscription(Activity activity, [C]SubscriptionParams subscriptionParams)
```

구독 상품 관리 메뉴로 이동합니다.

| **Parameters:**    |                                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| activity           | Activity: 구독 상풍 관리 메뉴로 이동하기 위해 activity를 참조합니다.                                                                        |
| subscriptionParams | <p><a href="subscriptionparams">SubscriptionParams</a> null 일 경우 관리 메뉴 화면으로<br>PurchaseData 가 있을 경우 상세 화면으로 이동합니다.</p> |

### getStoreInfoAsync <a href="#id-c-purchaseclient-getstoreinfoasync" id="id-c-purchaseclient-getstoreinfoasync"></a>

```
@AnyThread
public abstract void getStoreInfoAsync(StoreInfoListener listener)
```

마켓 구분코드를 가져옵니다.\
이 값은 server to server API를 사용할 때 필요한 값 입니다.

| **Parameters:** |                                                                        |
| --------------- | ---------------------------------------------------------------------- |
| listener        | [StoreInfoListener](../interfaces/storeinfolistener) 마켓 구분코드를 전달 받습니다. |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/purchaseclient.builder.md
# PurchaseClient.Builder

```
public static final class PurchaseClient.Builder
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.PurchaseClient.Builder
```

[PurchaseClient](purchaseclient) 인스턴스를 만들기 위한 빌더입니다.

## Public Methods <a href="#id-c-purchaseclient.builder-publicmethods" id="id-c-purchaseclient.builder-publicmethods"></a>

***

### setBase64PublicKey <a href="#id-c-purchaseclient.builder-setbase64publickey" id="id-c-purchaseclient.builder-setbase64publickey"></a>

```
@UiThread
public PurchaseClient.Builder setBase64PublicKey(String base64PublicKey)
```

구매정보의 유효성 체크를 위해 개발자센터에서 발급된 public key 를 지정하세요.

| **Parameters:**                   |                        |
| --------------------------------- | ---------------------- |
| base64PublicKey                   | base64EncodedPublicKey |
| **Returns**                       |                        |
| <p>PurchaseClient.Builder<br></p> |                        |

### setListener <a href="#id-c-purchaseclient.builder-setlistener" id="id-c-purchaseclient.builder-setlistener"></a>

```
@UiThread
public PurchaseClient.Builder setListener([I]PurchasesUpdatedListener listener)
```

onPurchasesUpdated 이벤트를 받을 수 있게 지정하세요.

| **Parameters:**                   |                                                                    |
| --------------------------------- | ------------------------------------------------------------------ |
| listener                          | [PurchasesUpdatedListener](../interfaces/purchasesupdatedlistener) |
| **Returns:**                      |                                                                    |
| <p>PurchaseClient.Builder<br></p> |                                                                    |

### build <a href="#id-c-purchaseclient.builder-build" id="id-c-purchaseclient.builder-build"></a>

```
@UiThread
public [C]PurchaseClient build()
```

PurchaseClient 인스턴스를 생성합니다.

| **Return:**                           |                                          |
| ------------------------------------- | ---------------------------------------- |
| [\[C\]PurchaseClient](purchaseclient) |                                          |
| **Throws:**                           |                                          |
| java.lang.IllegalArgumentException    | context or listener 가 정상적이지 않을 경우 발생합니다. |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/purchasedata.md
# PurchaseData

```
public class PurchaseData extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.PurchaseData
```

인앱결제에 대한 구매 데이터를 나타냅니다.

## Summary <a href="#id-c-purchasedata-summary" id="id-c-purchasedata-summary"></a>

***

| Nested classes |                                                                                    |
| -------------- | ---------------------------------------------------------------------------------- |
| @interface     | [\[A\]PurchaseData.AcknowledgeState](../annotations/purchasedata.acknowledgestate) |
| @interface     | [\[A\]PurchaseData.PurchaseState](../annotations/purchasedata.purchasestate)       |
| @interface     | [\[A\]PurchaseData.RecurringState](../annotations/purchasedata.recurringstate)     |

<table><thead><tr><th>Public constructors</th></tr></thead><tbody><tr><td><pre><code>PurchaseData(String jsonPurchaseData)
</code></pre></td></tr><tr><td><pre><code>PurchaseData(String jsonPurchaseData, String signature)
</code></pre></td></tr></tbody></table>

## Public methods <a href="#id-c-purchasedata-publicmethods" id="id-c-purchasedata-publicmethods"></a>

***

### getOrderId <a href="#id-c-purchasedata-getorderid" id="id-c-purchasedata-getorderid"></a>

```
String getOrderId()
```

구매에 대한 주문 아이디를 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getPackageName <a href="#id-c-purchasedata-getpackagename" id="id-c-purchasedata-getpackagename"></a>

```
String getPackageName()
```

구매를 시작한 어플리케이션의 packageName을 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getProductId <a href="#id-c-purchasedata-getproductid" id="id-c-purchasedata-getproductid"></a>

```
String getProductId()
```

구매한 인앱상품의 아이디를 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getPurchaseTime <a href="#id-c-purchasedata-getpurchasetime" id="id-c-purchasedata-getpurchasetime"></a>

```
long getPurchaseTime()
```

인앱상품을 구매한 시간을 밀리초 단위로 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| long         | <p><br></p> |

### isAcknowledged <a href="#id-c-purchasedata-isacknowledged" id="id-c-purchasedata-isacknowledged"></a>

```
boolean isAcknowledged()
```

구매가 확인(acknowledge) 되었는지 여부를 나타냅니다.

| **Returns:** |             |
| ------------ | ----------- |
| boolean      | <p><br></p> |

### getDeveloperPayload <a href="#id-c-purchasedata-getdeveloperpayload" id="id-c-purchasedata-getdeveloperpayload"></a>

```
String getDeveloperPayload()
```

구매의 확인(acknowledge)이나 소비(consume)를 할 때 지정했던 개발사의 페이로드를 전달합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getPurchaseId <a href="#id-c-purchasedata-getpurchaseid" id="id-c-purchasedata-getpurchaseid"></a>

```
@Deprecated
String getPurchaseId()
```

구매한 데이터를 고유하게 식별하는 아이디를 반환합니다.\
API v6 에서는 사용되지 않습니다.

<table data-header-hidden><thead><tr><th></th><th></th></tr></thead><tbody><tr><td><strong>Returns:</strong></td><td></td></tr><tr><td>String</td><td><br></td></tr><tr><td><strong>See Also:</strong></td><td></td></tr><tr><td><pre><code>getPurchaseToken()
</code></pre></td><td></td></tr></tbody></table>

### getPurchaseToken <a href="#id-c-purchasedata-getpurchasetoken" id="id-c-purchasedata-getpurchasetoken"></a>

```
String getPurchaseToken()
```

구매한 데이터를 고유하게 식별하는 토큰을 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getPurchaseState <a href="#id-c-purchasedata-getpurchasestate" id="id-c-purchasedata-getpurchasestate"></a>

```
int getPurchaseState()
```

구매 상태를 나타내는 값으로 \[A]PurchaseData.PurchaseState 중 하나를 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| int          | <p><br></p> |

### getRecurringState <a href="#id-c-purchasedata-getrecurringstate" id="id-c-purchasedata-getrecurringstate"></a>

```
int getRecurringState()
```

월정액 상품의 상태를 나타내는 값으로 [\[A\]PurchaseData.RecurringState](../annotations/purchasedata.recurringstate) 중 하나를 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| int          | <p><br></p> |

### getQuantity <a href="#id-c-purchasedata-getquantity" id="id-c-purchasedata-getquantity"></a>

```
int getQuantity()
```

상품의 수량을 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| int          | <p><br></p> |

### getSignature <a href="#id-c-purchasedata-getsignature" id="id-c-purchasedata-getsignature"></a>

```
String getSignature()
```

개발사의 개인 키로 서명된 구매 데이터의 서명이 포함된 문자열을 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getOriginalJson <a href="#id-c-purchasedata-getoriginaljson" id="id-c-purchasedata-getoriginaljson"></a>

```
String getOriginalJson()
```

구매 데이터에 대한 세부정보가 포함된 JSON 형식의 문자열을 반환 합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/purchaseflowparams.md
# PurchaseFlowParams

```
public class PurchaseFlowParams extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.PurchaseFlowParams
```

인앱 상품 구매를 하기 위한 파라미터 입니다.

**See Also:**

```
PurchaseClient.launchPurchaseFlow(Activity, [C]PurchaseFlowParams)
```

## Summary <a href="#id-c-purchaseflowparams-summary" id="id-c-purchaseflowparams-summary"></a>

***

| Nested classes |                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| class          | <p><a href="purchaseflowparams.builder">[C]PurchaseFlowParams.Builder</a></p><p>PurchaseFlowParams의 인스턴스를 쉽게 만들기 위한 빌더입니다.</p> |

## Public methods <a href="#id-c-purchaseflowparams-publicmethods" id="id-c-purchaseflowparams-publicmethods"></a>

***

### getProductId <a href="#id-c-purchaseflowparams-getproductid" id="id-c-purchaseflowparams-getproductid"></a>

```
String getProductId()
```

구매할 인앱 상품의 아이디 입니다.\


| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getProductType <a href="#id-c-purchaseflowparams-getproducttype" id="id-c-purchaseflowparams-getproducttype"></a>

```
String getProductType()
```

구매할 인앱 상품의 타입 입니다.\


| **Returns:** |                                                                              |
| ------------ | ---------------------------------------------------------------------------- |
| String       | [\[A\]PurchaseClient.ProductType](../annotations/purchaseclient.producttype) |

### getProductName <a href="#id-c-purchaseflowparams-getproductname" id="id-c-purchaseflowparams-getproductname"></a>

```
String getProductName()
```

등록된 상품의 이름이 아닌 구매 당시의 노출될 상품의 이름 입니다. null일 경우 등록된 상품명이 구매페이지에서 등록된 상품명이 노출됩니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getGameUserId <a href="#id-c-purchaseflowparams-getgameuserid" id="id-c-purchaseflowparams-getgameuserid"></a>

```
String getGameUserId()
```

개발사에서 확인 가능한 사용자의 아이디 입니다.\


| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### isPromotionApplicable <a href="#id-c-purchaseflowparams-ispromotionapplicable" id="id-c-purchaseflowparams-ispromotionapplicable"></a>

```
boolean isPromotionApplicable()
```

프로모션 가능 여부를 나타냅니다.\


| **Returns:** |             |
| ------------ | ----------- |
| boolean      | <p><br></p> |

### getDeveloperPayload <a href="#id-c-purchaseflowparams-getdeveloperpayload" id="id-c-purchaseflowparams-getdeveloperpayload"></a>

```
String getDeveloperPayload()
```

개발사에서 지정한 페이로드를 반환합니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getOldPurchaseToken <a href="#id-c-purchaseflowparams-getoldpurchasetoken" id="id-c-purchaseflowparams-getoldpurchasetoken"></a>

```
String getOldPurchaseToken()
```

변경할 정기 결제 상품의 구매 토큰입니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### getProrationMode() <a href="#id-c-purchaseflowparams-getprorationmode" id="id-c-purchaseflowparams-getprorationmode"></a>

```
int getProrationMode()
```

업그레이드 또는 다운그레이드를 위한 비례 배분 모드입니다.

| **Returns:** |                                  |
| ------------ | -------------------------------- |
| int          | PurchaseFlowParams.ProrationMode |

### getQuantity() <a href="#id-c-purchaseflowparams-getquantity" id="id-c-purchaseflowparams-getquantity"></a>

```
int getQuantity()
```

상품의 수량을 나타냅니다.

| **Returns:** |             |
| ------------ | ----------- |
| int          | <p><br></p> |

### newBuilder <a href="#id-c-purchaseflowparams-newbuilder" id="id-c-purchaseflowparams-newbuilder"></a>

```
 [C]PurchaseFlowParams.Builder newBuilder()
```

PurchaseFlowParams의 인스턴스를 만들기 위한 빌더를 생성합니다.

| **Returns:**                                                  |             |
| ------------------------------------------------------------- | ----------- |
| [\[C\]PurchaseFlowParams.Builder](purchaseflowparams.builder) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/purchaseflowparams.builder.md
# PurchaseFlowParams.Builder

```
public final class PurchaseFlowParams.Builder extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.PurchaseFlowParams.Builder
```

[PurchaseFlowParams](purchaseflowparams)의 인스턴스를 쉽게 만들기 위한 빌더입니다.

## Public methods <a href="#id-c-purchaseflowparams.builder-publicmethods" id="id-c-purchaseflowparams.builder-publicmethods"></a>

***

### setProductId <a href="#id-c-purchaseflowparams.builder-setproductid" id="id-c-purchaseflowparams.builder-setproductid"></a>

```
PurchaseFlowParams.Builder setProductId(String productId)
```

| **Parameters:**            |             |
| -------------------------- | ----------- |
| productId                  | 인앱 상품의 아이디  |
| **Returns:**               |             |
| PurchaseFlowParams.Builder | <p><br></p> |

### setProductType <a href="#id-c-purchaseflowparams.builder-setproducttype" id="id-c-purchaseflowparams.builder-setproducttype"></a>

```
PurchaseFlowParams.Builder setProductType(String productType)
```

| **Parameters:**            |                                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| productType                | <p>인앱 상품의 상품 타입</p><p><a href="../annotations/purchaseclient.producttype">PurchaseClient.ProductType</a></p> |
| **Returns:**               |                                                                                                              |
| PurchaseFlowParams.Builder | <p><br></p>                                                                                                  |

### setProductName <a href="#id-c-purchaseflowparams.builder-setproductname" id="id-c-purchaseflowparams.builder-setproductname"></a>

```
PurchaseFlowParams.Builder setProductType(String productName)
```

| **Parameters:**            |                                 |
| -------------------------- | ------------------------------- |
| productName                | 등록된 상품의 이름이 아닌구매 당시의 노출될 상품의 이름 |
| **Returns:**               |                                 |
| PurchaseFlowParams.Builder | <p><br></p>                     |

### setDeveloperPayload <a href="#id-c-purchaseflowparams.builder-setdeveloperpayload" id="id-c-purchaseflowparams.builder-setdeveloperpayload"></a>

```
PurchaseFlowParams.Builder setDeveloperPayload(String developerPayload)
```

| **Parameters:**            |             |
| -------------------------- | ----------- |
| developerPayload           | 개발사의 페이로드   |
| **Returns:**               |             |
| PurchaseFlowParams.Builder | <p><br></p> |

### setGameUserId <a href="#id-c-purchaseflowparams.builder-setgameuserid" id="id-c-purchaseflowparams.builder-setgameuserid"></a>

```
PurchaseFlowParams.Builder setGameUserId(String gameUserId)
```

| **Parameters:**            |                                  |
| -------------------------- | -------------------------------- |
| gameUserId                 | <p>개발사에서 확인 가능한 사용자의 아이디<br></p> |
| **Returns:**               |                                  |
| PurchaseFlowParams.Builder | <p><br></p>                      |

### setPromotionApplicable <a href="#id-c-purchaseflowparams.builder-setpromotionapplicable" id="id-c-purchaseflowparams.builder-setpromotionapplicable"></a>

```
PurchaseFlowParams.Builder setPromotionApplicable(boolean promotion)
```

| **Parameters:**            |                       |
| -------------------------- | --------------------- |
| promotion                  | <p>프로모션 가능 여부<br></p> |
| **Returns:**               |                       |
| PurchaseFlowParams.Builder | <p><br></p>           |

### setQuantity <a href="#id-c-purchaseflowparams.builder-setquantity" id="id-c-purchaseflowparams.builder-setquantity"></a>

```
PurchaseFlowParams.Builder setQuantity(int quantity)
```

| **Parameters:**            |             |
| -------------------------- | ----------- |
| quantity                   | 삼품의 수량      |
| **Returns:**               |             |
| PurchaseFlowParams.Builder | <p><br></p> |

### setSubscriptionUpdateParams <a href="#id-c-purchaseflowparams.builder-setsubscriptionupdateparams" id="id-c-purchaseflowparams.builder-setsubscriptionupdateparams"></a>

```
PurchaseFlowParams.Builder setSubscriptionUpdateParams(SubscriptionUpdateParams params)
```

| **Parameters:**            |                                                                                               |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| params                     | <p>정기 결제 상품의 업그레이드 또는 다운그레이드를 위한 모드<br></p><p>PurchaseFlowParams.SubscriptionUpdateParams</p> |
| **Returns:**               |                                                                                               |
| PurchaseFlowParams.Builder | <p><br></p>                                                                                   |

### build <a href="#id-c-purchaseflowparams.builder-build" id="id-c-purchaseflowparams.builder-build"></a>

```
 [C]PurchaseFlowParams build()
```

PurchaseFlowParams의 인스턴스를 생성합니다.

| **Returns:**                             |             |
| ---------------------------------------- | ----------- |
| [PurchaseFlowParams](purchaseflowparams) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/recurringproductparams.md
# RecurringProductParams

```
public class RecurringProductParams extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.RecurringProductParams
```

\


월정액 상품의 상태를 변경하기 위한 파라미터 입니다.

**See Also:**

[PurchaseClient.manageRecurringProductAsync](purchaseclient)(RecurringProductParams, [\[I\]RecurringProductListener](../interfaces/recurringproductlistener))\


## Summary <a href="#id-c-recurringproductparams-summary" id="id-c-recurringproductparams-summary"></a>

***

| Nested classes |                                                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| class          | <p><a href="recurringproductparams.builder">RecurringProductParams.Builder</a></p><p>RecurringProductParams의 인스턴스를 쉽게 만들기 위한 빌더입니다.</p> |

## Public methods <a href="#id-c-recurringproductparams-publicmethods" id="id-c-recurringproductparams-publicmethods"></a>

***

### getPurchaseData <a href="#id-c-recurringproductparams-getpurchasedata" id="id-c-recurringproductparams-getpurchasedata"></a>

```
PurchaseData getPurchaseData()
```

월정액 상품의 구매 데이터를 반환합니다.

| **Returns:**                 |             |
| ---------------------------- | ----------- |
| [PurchaseData](purchasedata) | <p><br></p> |

### getRecurringAction <a href="#id-c-recurringproductparams-getrecurringaction" id="id-c-recurringproductparams-getrecurringaction"></a>

```
String getRecurringAction()
```

월정액 상품의 변경하고 싶은 상태값을 반환합니다.

| **Returns:** |                                                                                 |
| ------------ | ------------------------------------------------------------------------------- |
| String       | [PurchaseClient.RecurringAction](../annotations/purchaseclient.recurringaction) |

### newBuilder <a href="#id-c-recurringproductparams-newbuilder" id="id-c-recurringproductparams-newbuilder"></a>

```
RecurringProductParams.Builder newBuilder()
```

RecurringProductParams의 인스턴스를 만들기 위한 빌더를 생성합니다.

| **Returns:**                                                     |             |
| ---------------------------------------------------------------- | ----------- |
| [RecurringProductParams.Builder](recurringproductparams.builder) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/recurringproductparams.builder.md
# RecurringProductParams.Builder

```
public final class RecurringProductParams.Builder extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.RecurringProductParams.Builder
```

[RecurringProductParams](recurringproductparams)의 인스턴스를 쉽게 만들기 위한 빌더입니다.

## Public methods <a href="#id-c-recurringproductparams.builder-publicmethods" id="id-c-recurringproductparams.builder-publicmethods"></a>

***

### setPurchaseData <a href="#id-c-recurringproductparams.builder-setpurchasedata" id="id-c-recurringproductparams.builder-setpurchasedata"></a>

```
RecurringProductParams.Builder setPurchaseData([C]PurchaseData purchaseData)
```

| **Parameters:**                |             |
| ------------------------------ | ----------- |
| purchaseData                   | <p><br></p> |
| **Returns:**                   |             |
| RecurringProductParams.Builder | <p><br></p> |

### setRecurringAction <a href="#id-c-recurringproductparams.builder-setrecurringaction" id="id-c-recurringproductparams.builder-setrecurringaction"></a>

```
RecurringProductParams.Builder setRecurringAction(String action)
```

<table data-header-hidden><thead><tr><th></th><th></th></tr></thead><tbody><tr><td><strong>Parameters:</strong></td><td></td></tr><tr><td><pre><code>action
</code></pre></td><td><p>월정액 상품의 상태를 변경하고 싶은 액션명 입니다.</p><p><a href="../annotations/purchaseclient.recurringaction">PurchaseClient.RecurringAction</a></p></td></tr><tr><td><strong>Returns:</strong></td><td></td></tr><tr><td>RecurringProductParams.Builder</td><td><br></td></tr></tbody></table>

### build <a href="#id-c-recurringproductparams.builder-build" id="id-c-recurringproductparams.builder-build"></a>

```
RecurringProductParams build()
```

RecurringProductParams의 인스턴스를 생성합니다.

| **Returns:**                                     |             |
| ------------------------------------------------ | ----------- |
| [RecurringProductParams](recurringproductparams) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/subscriptionparams.md
# SubscriptionParams

```
public class SubscriptionParams extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.SubscriptionParams
```

\


구독한 상품의 상세 관리 화면으로 이동하기 위한 파라미터 입니다.

**See Also:**

[PurchaseClient.launchManageSubscription](purchaseclient)(Activity activity, [SubscriptionParams](subscriptionparams) subscriptionParams)

## Summary <a href="#id-c-subscriptionparams-summary" id="id-c-subscriptionparams-summary"></a>

***

| Nested classes |                                                                                                                             |
| -------------- | --------------------------------------------------------------------------------------------------------------------------- |
| class          | <p><a href="subscriptionparams.builder">SubscriptionParams.Builder</a></p><p>SubscriptionParams의 인스턴스를 쉽게 만들기 위한 빌더입니다.</p> |

## Public methods <a href="#id-c-subscriptionparams-publicmethods" id="id-c-subscriptionparams-publicmethods"></a>

***

### getPurchaseData <a href="#id-c-subscriptionparams-getpurchasedata" id="id-c-subscriptionparams-getpurchasedata"></a>

```
PurchaseData getPurchaseData()
```

인앱 상품의 구매 데이터를 반환합니다.

| **Returns:**                 |             |
| ---------------------------- | ----------- |
| [PurchaseData](purchasedata) | <p><br></p> |

### newBuilder <a href="#id-c-subscriptionparams-newbuilder" id="id-c-subscriptionparams-newbuilder"></a>

```
SubscriptionParams.Builder newBuilder()
```

SubscriptionParams의 인스턴스를 만들기 위한 빌더를 생성합니다.

| **Returns:**                                             |             |
| -------------------------------------------------------- | ----------- |
| [SubscriptionParams.Builder](subscriptionparams.builder) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/subscriptionparams.builder.md
# SubscriptionParams.Builder

```
public final class SubscriptionParams.Builder extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.SubscriptionParams.Builder
```

\


[SubscriptionParams](acknowledgeparams)의 인스턴스를 쉽게 만들기 위한 빌더입니다.

## Public methods <a href="#id-c-subscriptionparams.builder-publicmethods" id="id-c-subscriptionparams.builder-publicmethods"></a>

***

### setPurchaseData <a href="#id-c-subscriptionparams.builder-setpurchasedata" id="id-c-subscriptionparams.builder-setpurchasedata"></a>

```
SubscriptionParams.Builder setPurchaseData([C]PurchaseData purchaseData)
```

| **Parameters:**            |             |
| -------------------------- | ----------- |
| purchaseData               | <p><br></p> |
| **Returns:**               |             |
| SubscriptionParams.Builder | <p><br></p> |

### build <a href="#id-c-subscriptionparams.builder-build" id="id-c-subscriptionparams.builder-build"></a>

```
SubscriptionParams build()
```

SubscriptionParams의 인스턴스를 생성합니다.

| **Returns:**                             |             |
| ---------------------------------------- | ----------- |
| [SubscriptionParams](subscriptionparams) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/subscriptionupdateparams.md
# SubscriptionUpdateParams

```
public class SubscriptionUdpateParams extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.PurchaseFlowParams.SubscriptionUpdateParams
```

인앱 상품 구매를 하기 위한 파라미터 입니다.

**See Also:**

```
PurchaseClient.launchPurchaseFlow(Activity, PurchaseFlowParams)
```

## Summary <a href="#id-c-subscriptionupdateparams-summary" id="id-c-subscriptionupdateparams-summary"></a>

***

| Nested classes |                                                                                                                                               |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| class          | <p><a href="subscriptionupdateparams.builder">SubscriptionUpdateParams.Builder</a></p><p>SubscriptionUpdateParams의 인스턴스를 쉽게 만들기 위한 빌더입니다.</p> |

## Public methods <a href="#id-c-subscriptionupdateparams-publicmethods" id="id-c-subscriptionupdateparams-publicmethods"></a>

***

### getProrationMode() <a href="#id-c-subscriptionupdateparams-getprorationmode" id="id-c-subscriptionupdateparams-getprorationmode"></a>

```
int getProrationMode()
```

정기 결제 상품의 업그레이드 또는 다운그레이드를 위한 비례 배분 모드 입니다.\


| **Returns:** |                                                                                     |
| ------------ | ----------------------------------------------------------------------------------- |
| int          | [PurchaseFlowParams.ProrationMode](../annotations/purchaseflowparams.prorationmode) |

### getOldPurchaseToken() <a href="#id-c-subscriptionupdateparams-getoldpurchasetoken" id="id-c-subscriptionupdateparams-getoldpurchasetoken"></a>

```
String getOldPurchaseToken()
```

구매한 정기 결제 상품의 구매 토큰 입니다.

| **Returns:** |             |
| ------------ | ----------- |
| String       | <p><br></p> |

### newBuilder <a href="#id-c-subscriptionupdateparams-newbuilder" id="id-c-subscriptionupdateparams-newbuilder"></a>

```
SubscriptionUpdateParams.Builder newBuilder()
```

SubscriptionUdpateParams의 인스턴스를 만들기 위한 빌더를 생성합니다.

| **Returns:**                                                         |             |
| -------------------------------------------------------------------- | ----------- |
| [SubscriptionUpdateParams.Builder](subscriptionupdateparams.builder) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/classes/subscriptionupdateparams.builder.md
# SubscriptionUpdateParams.Builder

```
public class SubscriptionUdpateParams extends Object
```

```
java.lang.Object
    ↳ com.gaa.sdk.iap.PurchaseFlowParams.SubscriptionUpdateParams.Builder
```

[SubscriptionUdpateParams](subscriptionupdateparams)의 인스턴스를 쉽게 만들기 위한 빌더입니다.

## Public methods <a href="#id-c-subscriptionupdateparams.builder-publicmethods" id="id-c-subscriptionupdateparams.builder-publicmethods"></a>

***

### setProrationMode() <a href="#id-c-subscriptionupdateparams.builder-setprorationmode" id="id-c-subscriptionupdateparams.builder-setprorationmode"></a>

```
SubscriptionUpdateParams.Builder setProrationMode(ProrationMode prorationMode)
```

| Parameters:                      |                                                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| prorationMode                    | <p>정기 결제 상품의 업그레이드 또는 다운그레이드를 위한 비례 배분 모드 입니다.</p><p><a href="../annotations/purchaseflowparams.prorationmode">ProrationMode</a></p> |
| **Returns:**                     |                                                                                                                                      |
| SubscriptionUpdateParams.Builder | <p><br></p>                                                                                                                          |

### setOldPurchaseToken() <a href="#id-c-subscriptionupdateparams.builder-setoldpurchasetoken" id="id-c-subscriptionupdateparams.builder-setoldpurchasetoken"></a>

```
SubscriptionUpdateParams.Builder setOldPurchaseToken()
```

| Parameters:                      |                          |
| -------------------------------- | ------------------------ |
| String                           | 구매한 정기 결제 상품의 구매 토큰 입니다. |
| **Returns:**                     |                          |
| SubscriptionUpdateParams.Builder | <p><br></p>              |

### newBuilder <a href="#id-c-subscriptionupdateparams.builder-newbuilder" id="id-c-subscriptionupdateparams.builder-newbuilder"></a>

```
SubscriptionUpdateParams newBuilder()
```

SubscriptionUdpateParams의 인스턴스를 생성합니다.

| **Returns:**                                         |             |
| ---------------------------------------------------- | ----------- |
| [SubscriptionUpdateParams](subscriptionupdateparams) | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces.md
# Interfaces

- [AcknowledgeListener](/dev/tools/billing/v21/references/interfaces/acknowledgelistener.md)
- [ConsumeListener](/dev/tools/billing/v21/references/interfaces/consumelistener.md)
- [IapResultListener](/dev/tools/billing/v21/references/interfaces/iapresultlistener.md)
- [ProductDetailsListener](/dev/tools/billing/v21/references/interfaces/productdetailslistener.md)
- [PurchaseClientStateListener](/dev/tools/billing/v21/references/interfaces/purchaseclientstatelistener.md)
- [PurchasesListener (deprecated)](/dev/tools/billing/v21/references/interfaces/purchaseslistener-deprecated.md)
- [PurchasesUpdatedListener](/dev/tools/billing/v21/references/interfaces/purchasesupdatedlistener.md)
- [QueryPurchasesListener](/dev/tools/billing/v21/references/interfaces/querypurchaseslistener.md)
- [RecurringProductListener](/dev/tools/billing/v21/references/interfaces/recurringproductlistener.md)
- [StoreInfoListener](/dev/tools/billing/v21/references/interfaces/storeinfolistener.md)


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces/acknowledgelistener.md
# AcknowledgeListener

```
public interface AcknowledgeListener
```

```
com.gaa.sdk.iap.AcknowledgeListener
```

구매확인이 완료되면 알려주는 리스너 입니다.

## Public methods <a href="#id-i-acknowledgelistener-publicmethods" id="id-i-acknowledgelistener-publicmethods"></a>

***

### onAcknowledgeResponse <a href="#id-i-acknowledgelistener-onacknowledgeresponse" id="id-i-acknowledgelistener-onacknowledgeresponse"></a>

```
void onAcknowledgeResponse(IapResult iapResult, PurchaseData purchaseData)
```

구매확인 작업이 완료 되었음을 알리기 위해 호출됩니다.

| **Parameters:** |             |
| --------------- | ----------- |
| iapResult       | 구매확인에 대한 결과 |
| purchaseData    | 구매 데이터      |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces/consumelistener.md
# ConsumeListener

```
public interface ConsumeListener
```

```
com.gaa.sdk.iap.ConsumeListener
```

소비가 완료되면 알려주는 리스너 입니다.

## Public methods <a href="#id-i-consumelistener-publicmethods" id="id-i-consumelistener-publicmethods"></a>

***

### onConsumeResponse <a href="#id-i-consumelistener-onconsumeresponse" id="id-i-consumelistener-onconsumeresponse"></a>

```
void onConsumeResponse(IapResult iapResult, PurchaseData purchaseData)
```

소비가 완료 되었음을 알리기 위해 호출됩니다.

| **Parameters:** |           |
| --------------- | --------- |
| iapResult       | 소비에 대한 결과 |
| purchaseData    | 구매 데이터    |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces/iapresultlistener.md
# IapResultListener

```
public interface IapResultListener
```

```
com.gaa.sdk.iap.IapResultListener
```

In-app Purchase API 응답에 대한 기본 리스너 입니다.

## Public methods <a href="#id-i-iapresultlistener-publicmethods" id="id-i-iapresultlistener-publicmethods"></a>

***

### onResponse <a href="#id-i-iapresultlistener-onresponse" id="id-i-iapresultlistener-onresponse"></a>

```
void onResponse( iapResult)
```

API 작업이 완료 되었음을 알리기 위해 호출됩니다.

| **Parameters:** |             |
| --------------- | ----------- |
| iapResult       | <p><br></p> |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces/productdetailslistener.md
# ProductDetailsListener

```
public interface ProductDetailsListener
```

```
com.gaa.sdk.iap.ProductDetailsListener
```

인앱 상품 상세정보 요청의 결과에 대한 리스너 입니다.

## Public methods <a href="#id-i-productdetailslistener-publicmethods" id="id-i-productdetailslistener-publicmethods"></a>

***

### onProductDetailsResponse <a href="#id-i-productdetailslistener-onproductdetailsresponse" id="id-i-productdetailslistener-onproductdetailsresponse"></a>

```
void onProductDetailsResponse( iapResult, List<> productDetailList)
```

인앱 상품 상세정보 요청작업이 완료 되었음을 알리기 위해 호출됩니다.

| **Parameters:**   |               |
| ----------------- | ------------- |
| iapResult         | 요청작업에 대한 결과   |
| productDetailList | 인앱상품 상세정보 리스트 |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces/purchaseclientstatelistener.md
# PurchaseClientStateListener

```
public interface PurchaseClientStateListener
```

```
com.gaa.sdk.iap.PurchaseClientStateListener
```

연결 설정 프로세스에 대한 리스너 입니다.

## Public methods <a href="#id-i-purchaseclientstatelistener-publicmethods" id="id-i-purchaseclientstatelistener-publicmethods"></a>

***

### onSetupFinished <a href="#id-i-purchaseclientstatelistener-onsetupfinished" id="id-i-purchaseclientstatelistener-onsetupfinished"></a>

```
void onSetupFinished(iapResult)
```

연결 설정이 완료 되었음을 알리기 위해 호출됩니다.\


| **Parameters:** |             |
| --------------- | ----------- |
| iapResult       | 요청작업에 대한 결과 |

### onServiceDisconnected <a href="#id-i-purchaseclientstatelistener-onservicedisconnected" id="id-i-purchaseclientstatelistener-onservicedisconnected"></a>

```
void onServiceDisconnected()
```

연결이 끊어졌음을 알리기 위해 호출됩니다.


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces/purchaseslistener-deprecated.md
# PurchasesListener (deprecated)

```
public interface PurchasesListener
```

```
com.gaa.sdk.iap.PurchasesListener
```

구매내역 요청이 완료되면 알려주는 리스너 입니다.

## Public methods <a href="#id-i-purchaseslistener-deprecated-publicmethods" id="id-i-purchaseslistener-deprecated-publicmethods"></a>

***

### onPurchasesResponse <a href="#id-i-purchaseslistener-deprecated-onpurchasesresponse" id="id-i-purchaseslistener-deprecated-onpurchasesresponse"></a>

```
void onPurchasesResponse(IapResult iapResult, List<PurchaseData> purchases)
```

구매내역 요청 작업이 완료 되었음을 알리기 위해 호출됩니다.

| **Parameters:** |                               |
| --------------- | ----------------------------- |
| iapResult       | 요청에 대한 결과                     |
| purchases       | 소비되지 않은 구매 데이터 내역 (월정액 상품 포함) |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces/purchasesupdatedlistener.md
# PurchasesUpdatedListener

```
public interface PurchasesUpdatedListener
```

```
com.gaa.sdk.iap.PurchasesUpdatedListener
```

구매를 성공적으로 완료되면 알려주는 리스너 입니다.\


## Public methods <a href="#id-i-purchasesupdatedlistener-publicmethods" id="id-i-purchasesupdatedlistener-publicmethods"></a>

***

### onPurchasesUpdated <a href="#id-i-purchasesupdatedlistener-onpurchasesupdated" id="id-i-purchasesupdatedlistener-onpurchasesupdated"></a>

```
void onPurchasesUpdated(IapResult iapResult, @Nullable List<PurchaseData> purchases)
```

구매가 정상적으로 완료 되었음을 알리기 위해 호출됩니다.\


| **Parameters:** |           |
| --------------- | --------- |
| iapResult       | 요청에 대한 결과 |
| purchases       | 구매한 데이터   |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces/querypurchaseslistener.md
# QueryPurchasesListener

```
public interface QueryPurchasesListener
```

```
com.gaa.sdk.iap.QueryPurchasesListener
```

구매내역 요청이 완료되면 알려주는 리스너 입니다.

## Public methods <a href="#id-i-querypurchaseslistener-publicmethods" id="id-i-querypurchaseslistener-publicmethods"></a>

***

### onPurchasesResponse <a href="#id-i-querypurchaseslistener-onpurchasesresponse" id="id-i-querypurchaseslistener-onpurchasesresponse"></a>

```
void onPurchasesResponse(IapResult iapResult, List<PurchaseData> purchases)
```

구매내역 요청 작업이 완료 되었음을 알리기 위해 호출됩니다.

| **Parameters:** |                               |
| --------------- | ----------------------------- |
| iapResult       | 요청에 대한 결과                     |
| purchases       | 소비되지 않은 구매 데이터 내역 (월정액 상품 포함) |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces/recurringproductlistener.md
# RecurringProductListener

```
public interface RecurringProductListener
```

```
com.gaa.sdk.iap.RecurringProductListener
```

월정액 상품의 상태변경이 완료되면 알려주는 리스너 입니다.

## Public methods <a href="#id-i-recurringproductlistener-publicmethods" id="id-i-recurringproductlistener-publicmethods"></a>

***

### onRecurringResponse <a href="#id-i-recurringproductlistener-onrecurringresponse" id="id-i-recurringproductlistener-onrecurringresponse"></a>

```
void onRecurringResponse(IapResult iapResult, PurchaseData purchaseData, String action)
```

월정액 상품의 상태변경 작업이 완료 되었음을 알리기 위해 호출됩니다.

| **Parameters:** |                                                                                                    |
| --------------- | -------------------------------------------------------------------------------------------------- |
| iapResult       | 구매확인에 대한 결과                                                                                        |
| purchaseData    | 요청할때 전달한 구매 데이터를 다시 반환                                                                             |
| action          | [PurchaseClient.RecurringAction](../annotations/purchaseclient.recurringaction) 요청할때 전달한 액션명 다시 반환 |


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/references/interfaces/storeinfolistener.md
# StoreInfoListener

```
public interface StoreInfoListener
```

```
com.gaa.sdk.iap.StoreInfoListener
```

마켓 구분코드 요청이 완료되면 알려주는 리스너 입니다.\


## Public methods <a href="#id-i-storeinfolistener-publicmethods" id="id-i-storeinfolistener-publicmethods"></a>

***

### onStoreInfoResponse <a href="#id-i-storeinfolistener-onstoreinforesponse" id="id-i-storeinfolistener-onstoreinforesponse"></a>

```
void onStoreInfoResponse(IapResult iapResult, String storeCode)
```

마켓 구분코드 요청 작업이 완료 되었음을 알리기 위해 호출됩니다.

<table data-header-hidden><thead><tr><th></th><th></th></tr></thead><tbody><tr><td><strong>Parameters:</strong></td><td></td></tr><tr><td>iapResult</td><td>요청에 대한 결과</td></tr><tr><td><pre><code>storeCode
</code></pre></td><td>Server to Server API 를 사용할 때 필요한 값입니다.</td></tr></tbody></table>


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/serverapi.md
# 06. 원스토어 인앱결제 서버 API (API V7)

## **개요**  <a href="#id-06.-api-apiv7" id="id-06.-api-apiv7"></a>

원스토어 인앱결제 서버 API란 원스토어에서 결제된 인앱 상품의 데이터를 조회하거나 결제 상태를 변경하기 위한 Open API를 말합니다.

해당 API를 사용하기 위해서는 OAuth 인증이 필요합니다.

### **HOST URL**  <a href="#id-06.-api-apiv7-hosturl" id="id-06.-api-apiv7-hosturl"></a>

본 문서에서 사용되는 Host Domain은 아래와 같습니다.

| 원스토어 환경 | Host Domain                                                    |
| ------- | -------------------------------------------------------------- |
| 검증(개발)  | <mark style="color:blue;">https://sbpp.onestore.net</mark>     |
| 상용      | <mark style="color:blue;">https://iap-apis.onestore.net</mark> |

{% hint style="info" %}
본 문서에서 사용하는 도메인은 모든 국가에서 공용으로 사용할 수 있는 도메인입니다.&#x20;

대한민국에서만 서비스 중인 기존 앱/게임의 도메인은 과거 도메인을 그대로 사용하시면 됩니다.&#x20;

(apis.onestore.co.kr)
{% endhint %}

### 마켓 구분 코드&#x20;

* Request header에 마켓 구분 코드 (x-market-code)가 추가 되었습니다.&#x20;
* 원스토어 글로벌 플랫폼에 서비스 되는 경우 마켓 구분 코드를 헤더에 추가하여 요청해야 합니다.&#x20;
* 마켓 구분 코드가 없는 경우에는 원스토어(기본 값)로 호출됩니다.&#x20;

| 마켓 구분 코드 | 서비스 대상      |
| -------- | ----------- |
| MKT\_ONE | 원스토어 (기본 값) |
| MKT\_GLB | 원스토어 글로벌    |

{% hint style="danger" %}
마켓 구분 코드에 따라 서버 API 응답으로 제공되는 시간의 기준이 다릅니다.&#x20;

MKT\_ONE : UTC+09

MKT\_GLB : UTC+00 &#x20;
{% endhint %}

## **원스토어 OAuth**  <a href="#id-06.-api-apiv7-oauth" id="id-06.-api-apiv7-oauth"></a>

### **OAuth 개요** <a href="#id-06.-api-apiv7-oauth" id="id-06.-api-apiv7-oauth"></a>

원스토어 서버 Open API를 연동하기 위해서는 OAuth 인증이 필요합니다.

* 원스토어 OAuth v2의 이해
  * AccessToken은 원스토어의 Server Open API를 통하여 발급받을 수 있는 값으로 원스토어에서 제공하는 Server Open API 호출시 인증값으로 사용됩니다.
  * AccessToken은 기본적으로 3600초의 유효기간이 있으며, 유효기간이 만료 되거나 600초 미만으로 남은 경우에 getAccessToken()을 호출하면 새로운 AccessToken이 발급됩니다.
    * 기존의 AccessToken도 유효기간이 끝날 때까지는 사용이 가능합니다.
    * 다수의 AccessToken이 발행되는 방식이므로 개발사의 서비스 인스턴스 별로 서로 다른 AccessToken을 취득하고 사용할 수 있는 형태가 됩니다. \
      \

*   일반적인 연동 흐름은 다음과 같습니다.\


    <figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2FrewUDEKtBdLoX8gRJs9L%2Fimage.png?alt=media&#x26;token=86dc5393-c873-48cb-af6a-1488ff07b0dd" alt=""><figcaption></figcaption></figure>
* AccessToken을 얻는 과정(1번)은 API 호출 시 인증 오류가 발생했을 때 호출하면 됩니다. \


원스토어의 인앱결제 서버 API의 호출을 위해서는 Authorization Bearer 스킴을 사용하며 호출 샘플은 아래와 같습니다.

```
GET /v7/apps/com.onestore.game.goindol/purchases/inapp/products/product01/SANDBOXT000120004476 HTTP/1.1
Host: sbpp.onestore.net
Authorization: Bearer 680b3621-1234-1234-1234-8adfaef561b4
```

Authorization 헤더에는 위 예제와 같이 Bearer + 공백 + AccessToken이 대소문자 구분하여 정확히 입력되어야 합니다.\
Bearer의 값은 getAccessToken()을 호출하여 발급받은 AccessToken 값입니다.

잘못된 예&#x20;

```
Authorization: 680b3621-1234-1234-1234-8adfaef561b4
Authorization: bearer 680b3621-1234-1234-1234-8adfaef561b4
Authorization: Bearer <680b3621-1234-1234-1234-8adfaef561b4>
Authorization:Bearer680b3621-1234-1234-1234-8adfaef561b4
```

\


### **OAuth API 상세** <a href="#id-06.-api-apiv7-oauthapi" id="id-06.-api-apiv7-oauthapi"></a>

**client\_id 및 client\_secret 확인**

Client\_id와 Client\_secret 값은 "라이선스 관리" 메뉴에서 확인할 수 있습니다.



### **AccessToken 발급**

* **URI :** /v7/oauth/token
* **Method:** POST
*   **Request Parameter:** Form 형식

    | Parameter Name | Description                        | Example                                      |
    | -------------- | ---------------------------------- | -------------------------------------------- |
    | client\_id     | 개발자센터에서 앱 등록 시 발급된 Client id 값     | 0000042301                                   |
    | client\_secret | 개발자센터에서 앱 등록 시 발급된 client secret 값 | vxIMAGcVz3DAx20uDBr/IDWNJAPNHFl7YruF4uxB6BI= |
    | grant\_type    | 고정값                                | client\_credentials                          |
*   **Request Header** :&#x20;

    | Parameter Name | Description                                                             | Example                                         |
    | -------------- | ----------------------------------------------------------------------- | ----------------------------------------------- |
    | Content-Type   | Http 요청시 Content Type으로 반드시 application/x-www-form-urlencoded 로 설정되어야 함 | Content-Type: application/x-www-form-urlencoded |
    | x-market-code  | 마켓 구분 코드                                                                | x-market-code: MKT\_GLB                         |
*   **Example**&#x20;

    ```
    POST /v7/oauth/token HTTP/1.1
    Host: apis.onestore.com
    Content-Type: application/x-www-form-urlencoded;charset=UTF-8
    x-market-code: MKT_GLB

    grant_type=client_credentials&client_id=com.onestore.game.goindol&client_secret=vxIMAGcVz3DAx20uDBr/IDWNJAPNHFl7YruF4uxB6BI=
    ```
*   **Response Body** : JSON 형식

    | Element Name  | Data Type | Data Size | Description                |
    | ------------- | --------- | --------- | -------------------------- |
    | client\_id    | String    | 255       | OAuth 인증 client\_id        |
    | access\_token | String    | 36        | AccessToken                |
    | token\_type   | String    | 6         | bearer 방식만 제공              |
    | expires\_in   | Integer   | 10        | token 만료기한, 단위 : 초(second) |
    | scope         | String    | 1024      | token 사용 범위                |
*   **Example :**&#x20;

    ```json
    {
        "client_id":"0000000001",
        "access_token":"680b3621-1234-1234-1234-8adfaef561b4",
        "token_type":"bearer",
        "expires_in":3010,
        "scope":"DEFAULT"
    }
    ```


*   **발급 예시** :&#x20;

    {% code overflow="wrap" %}
    ```bash
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
    {% endcode %}



검증(개발) 환경과 상용환경의 AccessToken은 독립적으로 관리되므로, 환경 별로 AccessToken을 분리하여 관리하셔야 합니다.

## **서버 API 상세**  <a href="#id-06.-api-apiv7-api" id="id-06.-api-apiv7-api"></a>

{% hint style="info" %}
2025년 3월 20일 개발자센터 개편의 영향으로 서버 API 요청 시 이용되던 packageName은 clientId로 변경되었습니다.&#x20;

* 3월 20일 이전 등록된 앱의 경우 클라이언트 ID 와 패키지명이 동일하므로, 변경에 대한 영향도가 없습니다.&#x20;
* 3월 20일 이후 등록된 앱의 경우 클라이언트 ID 가 PID와 동일하므로 참고 부탁 드립니다.&#x20;
{% endhint %}

### getPurchaseDetails (구매상품 상세조회) <a href="#id-06.-api-apiv7-getpurchasedetails" id="id-06.-api-apiv7-getpurchasedetails"></a>

* **Desc :** 구매한 원스토어 관리형 상품의 상세정보를 조회합니다. 구매완료 시 전달받은 원스토어 purchaseToken(구매 토큰)을 이용하여 조회하여야 합니다. 월정액 상품의 purchaseToken으로 조회 시, 조회 실패(NoSuchData) 응답이 전달됩니다.
* **URI :** /v7/apps/{clientId}/purchases/inapp/products/{productId}/{purchaseToken}
* **Method :** GET
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  * String productId : 상품 ID (Data Size : 150)
  * String purchaseToken : 구매 토큰 (Data Size : 20)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
* **Request Body :** 없음
*   **Response Body** : JSON 형식

    | Element Name     | Data Type | Data Size | Description                                    |
    | ---------------- | --------- | --------- | ---------------------------------------------- |
    | consumptionState | Integer   | 1         | 구매한 상품의 소비상태 (0: 미소비, 1: 소비)                   |
    | developerPayload | String    | 200       | 개발사가 제공한 결제 고유 식별자                             |
    | purchaseState    | Integer   | 1         | 구매상태 ( 0: 구매완료, 1: 취소완료)                       |
    | purchaseTime     | Long      | 13        | 구매시간 (ms)                                      |
    | purchaseId       | String    | 20        | 구매 ID                                          |
    | acknowledgeState | Integer   | 1         | 구매확인 상태( 0: Not Acknowledged, 1: Acknowledged) |
    | quantity         | Integer   | 2         | 구매 수량                                          |
*   **Example:**

    ```
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
    ```

### getRecurringPurchaseDetails (월정액 상품 구매 상세조회)  <a href="#id-06.-api-apiv7-getrecurringpurchasedetails" id="id-06.-api-apiv7-getrecurringpurchasedetails"></a>

* **Desc :** 구매한 원스토어 월정액 상품의 자동결제 상태와 마지막 구매상태의 상세정보를 조회합니다. 구매완료 시 전달받은 원스토어 purchaseToken(구매 토큰)을 이용하여 조회하여야 합니다. 관리형 상품의 purchaseToken으로 조회 시, 조회 실패(NoSuchData) 응답이 전달됩니다.
* **URI :** /v7/apps/{clientId}/purchases/auto/products/{productId}/{purchaseToken}
* **Method :** GET
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  * String productId : 상품 ID (Data Size : 150)
  * String purchaseToken : 구매 토큰 (Data Size : 20)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
* **Request Body :** 없음
*   **Response Body** : JSON 형식

    | Element Name      | Data Type | Data Size | Description                                            |
    | ----------------- | --------- | --------- | ------------------------------------------------------ |
    | startTime         | Long      | 13        | 구매한 상품의 이용시작시간(ms)                                     |
    | expiryTime        | Long      | 13        | 구매한 상품의 이용종료시간(ms)                                     |
    | nextPaymentTime   | Long      | 13        | 다음 자동결제시간                                              |
    | autoRenewing      | boolean   | -         | 이용종료시간(expiryTime)가 지난 후 자동결제 여부                       |
    | cancelReason      | Integer   | 1         | 해지사유 (0 : 고객요청에 의한 해지, 1 : 기타 시스템 처리로 인한 해지 )          |
    | cancelledTime     | Long      | 13        | 해지시간(ms)                                               |
    | acknowledgeState  | Integer   | 1         | 월정액 상품의 구매확인 상태( 0: Not Acknowledged, 1: Acknowledged) |
    | lastPurchaseId    | String    | 20        | 마지막에 자동결제된 구매ID                                        |
    | lastPurchaseState | Integer   | 1         | 마지막에 자동결제된 구매상태 (0 : 구매완료, 1 : 취소완료)                   |
*   **Example :**&#x20;

    ```json
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
    ```

    고객이 월정액 상품에 대하여 컨텐츠 사용 권한이 있는지 아래의 조건으로 판단이 가능합니다.

    * 현재시간이 expiryTime(구매한 상품의 이용종료시간)보다 작거나 같고, lastPurchaseState(마지막에 자동결제된 구매상태)가 0(구매완료) 상태인경우\
      Ex) expiryTime >= 현재시간 AND lastPurchaseState == 0

### acknowledgePurchase (구매상품 확인)  <a href="#id-06.-api-apiv7-acknowledgepurchase" id="id-06.-api-apiv7-acknowledgepurchase"></a>

* **Desc :** 구매한 관리형 또는 월정액 상품을 구매확인 상태로 변경한다.
* **URI :** /v7/apps/{clientId}/purchases/all/products/{productId}/{purchaseToken}/acknowledge
* **Method :** POST
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  * String productId : 상품 ID (Data Size : 150)
  * String purchaseToken : 구매 토큰 (Data Size : 20)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
*   **Request Body** : JSON 형식

    | Element Name     | Data Type | Required | Description |
    | ---------------- | --------- | -------- | ----------- |
    | developerPayload | String    | false    | <p><br></p> |
*   **Example :**

    ```json
    {
        "developerPayload": "your payload"
    }
    ```
*   **Response Body :** JSON 형식

    API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.

    | Element Name | Data Type | Data Size | Description |
    | ------------ | --------- | --------- | ----------- |
    | code         | String    | -         | 응답 코드       |
    | message      | String    | -         | 응답 메시지      |
    | result       | Object    | -         | <p><br></p> |
*   **Example :**&#x20;

    ```json
    HTTP/1.1 200 OK
    Content-type: application/json;charset=UTF-8
    {
        "result" : {
            "code" : "Success",
            "message" : "Request has been completed successfully."
        }
    }
    ```



원스토어는 3일 이내에 acknowledgePurchase API가 호출되지 않은 구매 건을 자동으로 취소합니다.

이에 따라 개발사는 반드시 해당 API를 호출해야 하며, SDK API 또는 서버 API를 통해 처리할 수 있습니다.

단, consumePurchase API가 호출된 구매 건은 acknowledge가 되었다고 판단하고 구매 취소를 하지 않습니다.

### consumePurchase (구매상품 소비) <a href="#id-06.-api-apiv7-consumepurchase" id="id-06.-api-apiv7-consumepurchase"></a>

* **Desc :** 구매한 관리형 인앱 상품을 소비 상태로 변경한다.
* **URI :** /v7/apps/{clientId}/purchases/inapp/products/{productId}/{purchaseToken}/consume
* **Method :** POST
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  * String productId : 상품 ID (Data Size : 150)
  * String purchaseToken : 구매 토큰 (Data Size : 20)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
*   **Request Body** : JSON 형식

    | Element Name     | Data Type | Required | Description |
    | ---------------- | --------- | -------- | ----------- |
    | developerPayload | String    | false    | <p><br></p> |
*   **Example :**

    ```json
    {
        "developerPayload": "your payload"
    }
    ```
*   **Response Body :** JSON 형식

    API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.

    | Element Name | Data Type | Data Size | Description |
    | ------------ | --------- | --------- | ----------- |
    | code         | String    | -         | 응답 코드       |
    | message      | String    | -         | 응답 메시지      |
    | result       | Object    | -         | <p><br></p> |
*   **Example :**&#x20;

    ```json
    HTTP/1.1 200 OK
    Content-type: application/json;charset=UTF-8
    {
        "result" : {
            "code" : "Success",
            "message" : "Request has been completed successfully."
        }
    }
    ```



### cancelRecurringPurchase (자동결제 해지요청)  <a href="#id-06.-api-apiv7-cancelrecurringpurchase" id="id-06.-api-apiv7-cancelrecurringpurchase"></a>

* **Desc :** 월정액 상품의 자동결제 해지를 요청한다. 구독형 상품의 purchaseToken이용 시, 조회 실패(NoSuchData) 응답이 전달됩니다.
* **URI :** /v7/apps/{clientId}/purchases/auto/products/{productId}/{purchaseToken}/cancel
* **Method :** POST
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  * String productId : 상품 ID (Data Size : 150)
  * String purchaseToken : 구매 토큰 (Data Size : 20)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
* **Request Body :** 없음
*   **Response Body :** JSON 형식

    API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.

    | Element Name | Data Type | Data Size | Description |
    | ------------ | --------- | --------- | ----------- |
    | code         | String    | -         | 응답 코드       |
    | message      | String    | -         | 응답 메시지      |
    | result       | Object    | -         | <p><br></p> |
*   **Example :**&#x20;

    ```json
    HTTP/1.1 200 OK
    Content-type: application/json;charset=UTF-8
    {
        "result" : {
            "code" : "Success",
            "message" : "Request has been completed successfully."
        }
    }
    ```

### reactiveRecurringPurchase (자동결제 해지 취소요청)  <a href="#id-06.-api-apiv7-reactiverecurringpurchase" id="id-06.-api-apiv7-reactiverecurringpurchase"></a>

* **Desc :** 월정액 상품의 자동결제 해지요청을 취소한다. 구독형 상품의 purchaseToken이용 시, 조회 실패(NoSuchData) 응답이 전달됩니다.
* **URI :** /v7/apps/{clientId}/purchases/auto/products/{productId}/{purchaseToken}/reactivate
* **Method :** POST
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  * String productId : 상품 ID (Data Size : 150)
  * String purchaseToken : 구매 토큰 (Data Size : 20)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
* **Request Body :** 없음
*   **Response Body :** JSON 형식

    API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.

    | Element Name | Data Type | Data Size | Description |
    | ------------ | --------- | --------- | ----------- |
    | code         | String    | -         | 응답 코드       |
    | message      | String    | -         | 응답 메시지      |
    | result       | Object    | -         | <p><br></p> |
*   **Example :**&#x20;

    ```json
    HTTP/1.1 200 OK
    Content-type: application/json;charset=UTF-8
    {
        "result" : {
            "code" : "Success",
            "message" : "Request has been completed successfully."
        }
    }
    ```

### getVoidedPurchases (구매취소내역 조회)  <a href="#id-06.-api-apiv7-getvoidedpurchases" id="id-06.-api-apiv7-getvoidedpurchases"></a>

* **Desc :** 구매취소내역을 조회한다.
* **URI :** /v7/apps/{clientId}/voided-purchases
* **Method :** GET
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
* **Request Parameter (Optional) :** Query String 형식\

  * String continuationKey : 구매취소 건이 많을 경우 원스토어 서버에서 이 값을 반환합니다. \
    응답에 continuationKey가 있을 경우, getVoidedPurchases를 다시 호출하면서 continuationKey를 전달하면 추가 구매취소 내역을 전달받을 수 있습니다. (Data Size : 41)
  * String startTime : 구매취소일시 검색 시작시간 (milliseconds). \
    현재시간기준 과거 1개월까지만 설정가능하며 startTime 단독으로 사용할 경우 endTime은 startTime기준 미래 1개월로 설정됩니다. (Data Size : 13)
  * String endTime :  구매취소일시 검색 종료시간 (milliseconds). \
    현재시간보다 클 수 없으며 endTime 단독으로 사용할 경우 startTime은 endTime기준 과거 1개월로 설정됩니다. (Data Size : 13)
  * unsigned integer maxResults : 최대조회건수 default 100 (Data Size : 3)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
* **Request Body :** 없음
*   **Response Body :**

    | Element Name        | Data Type   | Data Size | Description                                                                         |
    | ------------------- | ----------- | --------- | ----------------------------------------------------------------------------------- |
    | continuationKey     | String      | 41        | 구매취소 건이 최대조회건수보다 많을 경우 반환. 이후 요청시 전달받은 continuationKey 를 셋팅하여 호출하면 이후 내역을 조회 할 수 있음 |
    | voidedPurchaseList  | <p><br></p> | -         | <p><br></p>                                                                         |
    | purchaseId          | String      | 20        | 구매 ID                                                                               |
    | purchaseTime        | Long        | 13        | 구매시간(ms)                                                                            |
    | voidedTime          | Long        | 13        | 구매취소시간(ms)                                                                          |
    | purchaseToken       | String      | 20        | 구매 토큰                                                                               |
    | marketCode          | String      | -         | 마켓 구분 코드                                                                            |
*   **Example :**&#x20;

    ```json
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
    ```

### getSubscriptionDetail (구독 상세 조회)  <a href="#id-06.-api-apiv7-getsubscriptiondetail" id="id-06.-api-apiv7-getsubscriptiondetail"></a>

* **Desc :** 구독 상세 내역을 조회한다.
* **URI :** /v7/apps/{clientId}/purchases/subscription/products/{productId}/{purchaseToken}
* **Method :** GET
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  * String productId : 상품 ID (Data Size : 150)
  * String purchaseToken : 구매 토큰 (Data Size : 20)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
* **Request Body :** 없음
*   **Response Body :**

    <table data-header-hidden><thead><tr><th width="205"></th><th></th><th></th><th></th></tr></thead><tbody><tr><td>Element Name</td><td>Data Type</td><td>Data Size</td><td>Description</td></tr><tr><td>acknowledgementState</td><td>Integer</td><td>1</td><td>구매확인 상태( 0: Not Acknowledged, 1: Acknowledged)</td></tr><tr><td>developerPayload</td><td>String</td><td>200</td><td>개발사가 제공한 결제 고유 식별자</td></tr><tr><td>autoRenewing</td><td>boolean</td><td>-</td><td><p>다음결제 갱신여부<br></p><ul><li>구독해지예약,해지,만료 : false</li><li>나머지 : true</li></ul></td></tr><tr><td>paymentState</td><td>Integer</td><td>1</td><td><p>구독결제 수신여부</p><ul><li>null: 만료 상태</li><li>0: 결제가 완료 되지 않은 상태</li><li>1: 결제요청 후 결제완료가 된 상태</li><li>2: 무료기간으로 처리된 상태</li><li>3: 상품 업/다운그레이드 시 DEFERRED옵션으로 처리된 상태</li></ul></td></tr><tr><td>priceAmount</td><td>String</td><td>-</td><td>구매금액</td></tr><tr><td>priceAmountMicros</td><td>Long</td><td>-</td><td>구매금액 * 100만</td></tr><tr><td>nextPriceAmount</td><td>String</td><td>-</td><td>다음구매금액</td></tr><tr><td>nextPriceAmountMicros</td><td>Long</td><td>-</td><td>다음구매금액 * 100만</td></tr><tr><td>nextPaymentTimeMillis</td><td>Long</td><td>13</td><td>다음구매시간(ms)</td></tr><tr><td>priceCurrencyCode</td><td>String</td><td>-</td><td>통화코드(KRW 고정)</td></tr><tr><td>countryCode</td><td>String</td><td>-</td><td>국가코드(KR 고정)</td></tr><tr><td>startTimeMillis</td><td>Long</td><td>13</td><td>구독 시작(첫 결제)시간(ms)</td></tr><tr><td>expiryTimeMillis</td><td>Long</td><td>13</td><td>구독 만료시간(ms)</td></tr><tr><td>pauseStartTimeMillis</td><td>Long</td><td>13</td><td>일시정지 시작일시(ms)-일시정지예약/일시정지 시에만 존재</td></tr><tr><td>pauseEndTimeMillis</td><td>Long</td><td>13</td><td>일시정지 시작일시(ms)-일시정지예약/일시정지 시에만 존재</td></tr><tr><td>autoResumeTimeMillis</td><td>Long</td><td>13</td><td><p>일시정지 후 재구독 시간</p><ul><li>정상 구독 중: null</li><li>일시정지 예약/확정/일시정지중: 다음결제일 + 일시정지일자 </li></ul></td></tr><tr><td>linkedPurchaseToken</td><td>String</td><td>20</td><td>구독상품 변경 시 이전 purchaseToken, 변경한 적이 없으면 null</td></tr><tr><td>lastPurchaseId</td><td>String</td><td>20</td><td>마지막 구매ID</td></tr><tr><td>cancelledTimeMillis<br></td><td>Long</td><td>13</td><td>구독 해지시간(ms)</td></tr><tr><td>cancelReason</td><td>Integer</td><td>1</td><td><p>구독 해지사유 </p><ul><li>0 : 고객요청</li><li>1 : 기타</li></ul></td></tr><tr><td>promotionPrice</td><td>Object</td><td>-</td><td>프로모션 금액 정보</td></tr><tr><td>promotionPrice.promotionPrice</td><td>String</td><td>-</td><td>프로모션 금액</td></tr><tr><td>promotionPrice.promotionPriceMicros</td><td>Long</td><td>-</td><td>프로모션 금액 정보 * 100만</td></tr><tr><td>promotionPrice.promotionPeriod</td><td>Int</td><td>-</td><td>프로모션 적용 회차</td></tr><tr><td>priceChange</td><td>Object</td><td>-</td><td>가격변동정보</td></tr><tr><td>priceChange.seq</td><td>Integer</td><td>-</td><td>가격변동 시퀀스 </td></tr><tr><td>priceChange.previousPrice</td><td>String</td><td>-</td><td>이전 가격 </td></tr><tr><td>priceChange.previousPriceMicros</td><td>Long</td><td>-</td><td>이전 가격 * 100만</td></tr><tr><td>priceChange.newPrice</td><td>String</td><td>-</td><td>신규 가격</td></tr><tr><td>priceChange.newPriceMicros</td><td>Long</td><td>-</td><td>신규 가격 * 100만</td></tr><tr><td>priceChange.applyTimeMillis</td><td>Long</td><td>13</td><td>적용일자(ms)</td></tr><tr><td>priceChange.agreement</td><td>Boolean</td><td>-</td><td>가격 변동 동의 여부</td></tr><tr><td>priceChange.agreementDueDateTimeMillis</td><td>Long</td><td>13</td><td><p>가격 변동 동의 만료 시간(ms)</p><p>(정책 부연 설명)</p><p>Value = 가격변경일 +7+30일<br>사용자는 동의만료일 이후 첫 자동결제 시점까지 동의가능함.</p></td></tr></tbody></table>
*   **Example :**&#x20;

    ```json
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
    ```

### cancelSubscription (구독결제 해지요청)  <a href="#id-06.-api-apiv7-cancelsubscription" id="id-06.-api-apiv7-cancelsubscription"></a>

* **Desc :** 구독형 상품의 자동결제 해지를 요청한다. 단, 요청 시점에 구독상태가 일시중지, 결제유예, 계정보류일 경우 즉시 해지를 요청한다.
* **URI :** /v7/apps/{clientId}/purchases/subscription/products/{productId}/{purchaseToken}/cancel
* **Method :** POST
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  * String productId : 상품 ID (Data Size : 150)
  * String purchaseToken : 구매 토큰 (Data Size : 20)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
* **Request Body :** 없음
*   **Response Body :** JSON 형식

    API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.

    | Element Name | Data Type | Data Size | Description |
    | ------------ | --------- | --------- | ----------- |
    | code         | String    | -         | 응답 코드       |
    | message      | String    | -         | 응답 메시지      |
    | result       | Object    | -         | <p><br></p> |
*   **Example :**&#x20;

    ```json
    HTTP/1.1 200 OK
    Content-type: application/json;charset=UTF-8
    {
        "result" : {
            "code" : "Success",
            "message" : "Request has been completed successfully."
        }
    }
    ```

### reactivateSubscription (구독결제 해지 취소요청)  <a href="#id-06.-api-apiv7-reactivatesubscription" id="id-06.-api-apiv7-reactivatesubscription"></a>

* **Desc :** 구독형 상품의 자동결제 해지요청을 취소한다. 단, 즉시해지된 경우 해지요청을 취소할 수 없다.
* **URI :** /v7/apps/{clientId}/purchases/subscription/products/{productId}/{purchaseToken}/reactivate
* **Method :** POST
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  * String productId : 상품 ID (Data Size : 150)
  * String purchaseToken : 구매 토큰 (Data Size : 20)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
* **Request Body :** 없음
*   **Response Body :** JSON 형식

    API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.

    | Element Name | Data Type | Data Size | Description |
    | ------------ | --------- | --------- | ----------- |
    | code         | String    | -         | 응답 코드       |
    | message      | String    | -         | 응답 메시지      |
    | result       | Object    | -         | <p><br></p> |
*   **Example :**&#x20;

    ```json
    HTTP/1.1 200 OK
    Content-type: application/json;charset=UTF-8
    {
        "result" : {
            "code" : "Success",
            "message" : "Request has been completed successfully."
        }
    }
    ```

### deferSubscription (구독결제 연기요청)  <a href="#id-06.-api-apiv7-defersubscription" id="id-06.-api-apiv7-defersubscription"></a>

* **Desc :** 구독형 상품의 다음 결제일을 연기한다.
* **URI :** /v7/apps/{clientId}/purchases/subscription/products/{productId}/{purchaseToken}/defer
* **Method :** POST
* **Request Parameter :** Path Variable 형식
  * String clientId : API를 호출하는 앱의 클라이언트 ID (Data Size : 128)
  * String productId : 상품 ID (Data Size : 150)
  * String purchaseToken : 구매 토큰 (Data Size : 20)
*   **Request Header:**&#x20;

    | Parameter Name | Data Type | Required | Description                             |
    | -------------- | --------- | -------- | --------------------------------------- |
    | Authorization  | String    | true     | Access Token API를 통해 발급받은 access\_token |
    | Content-Type   | String    | true     | application/json                        |
    | x-market-code  | String    | false    | 마켓 구분 코드                                |
*   **Example**&#x20;

    ```java
    Request.setHeader("Authorization", "Bearer 680b3621-1234-1234-1234-8adfaef561b4");
    Request.setHeader("Content-Type", "application/json");
    Request.setHeader("x-market-code", "MKT_GLB");
    ```
*   **Request Body :** JSON 형식

    | Parameter Name | Data Type | Required | Description                                                       |
    | -------------- | --------- | -------- | ----------------------------------------------------------------- |
    | deferPeriod    | Integer   | true     | <p>연기 기간</p><ul><li>상용: 일 단위(1~365일)</li><li>샌드박스: 분 단위</li></ul> |
*   **Example :**&#x20;

    ```json
    Content-type: application/json;charset=UTF-8
    {
    	"deferPeriod" : 3
    }
    ```
*   **Response Body :** JSON 형식

    API 처리 성공 시 처리완료를 보다 직관적으로 판단할 수 있도록 아래 형식의 응답를 리턴합니다. 단, API 처리 실패 시에는 표준오류응답을 리턴합니다.

    | Element Name | Data Type | Data Size | Description |
    | ------------ | --------- | --------- | ----------- |
    | code         | String    | -         | 응답 코드       |
    | message      | String    | -         | 응답 메시지      |
    | result       | Object    | -         | <p><br></p> |
*   **Example :**&#x20;

    ```json
    HTTP/1.1 200 OK
    Content-type: application/json;charset=UTF-8
    {
        "result" : {
            "code" : "Success",
            "message" : "Request has been completed successfully."
        }
    }
    ```

## 표준응답규격 <a href="#id-06.-api-apiv7" id="id-06.-api-apiv7"></a>

### 표준응답코드 <a href="#id-06.-api-apiv7" id="id-06.-api-apiv7"></a>

| Code                       | Message                                                                               | Description                              | HTTP Status Code                      | 대상 API                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| AccessBlocked              | The request was blocked.                                                              | 요청이 차단되었습니다.                             | 403 - Fobidden                        | 공통                                                                                                      |
| AccessTokenExpired         | Access token has expired.                                                             | Access 토큰이 만료되었습니다.                      | 401 - Unauthorized                    | 공통                                                                                                      |
| BadRequest                 | The request are invalid.                                                              | 잘못된 요청입니다.                               | 400 - Bad Request                     | 공통                                                                                                      |
| DeveloperPayloadNotMatch   | The request developerPayload does not match the value passed in the purchase request. | 구매요청 시 전달된 developerPayload값과 일치하지 않습니다. | 400 - Bad Request                     | <p>acknowledgePurchase<br>consumePurchase</p>                                                           |
| InternalError              | An undefined error has occurred.                                                      | 정의되지 않은 오류가 발생하였습니다.                     | 500 - Internal Server Error           | 공통                                                                                                      |
| InvalidAccessToken         | Access token is invalid.                                                              | Access 토큰이 유효하지 않습니다.                    | 401 - Unauthorized                    | 공통                                                                                                      |
| InvalidAuthorizationHeader | Authorization header is invalid.                                                      | Authorization 헤더의 값이 유효하지 않습니다.          | 400 - Bad Request                     | 공통                                                                                                      |
| InvalidConsumeState        | The purchase consumption status cannot be changed or has already been changed.        | 소비상태 변경이 불가하거나, 이미 변경완료 되었습니다.           | 409 - Conflict                        | consumePurchase                                                                                         |
| InvalidContentType         | The request content-type is invalid.                                                  | 잘못된 Content Type 입니다.                    | 415 - Unsupported Media Type          | 공통                                                                                                      |
| InvalidPurchaseState       | Purchase history does not exist or is not completed.                                  | 구매내역이 존재하지 않거나, 구매완료 상태가 아닙니다.           | 409 - Conflict                        | <p>acknowledgePurchase<br>consumePurchase</p>                                                           |
| InvalidRequest             | Request parameters are invalid. \[ field1, field2, ... ]                              | 입력값이 유효하지 않습니다. \[ field1, field2, ... ] | 400 - Bad Request                     | 공통                                                                                                      |
| MethodNotAllowed           | HTTP method not supported.                                                            | 지원하지 않는 HTTP Method 입니다.                 | 405 - Method Not Allowed              | 공통                                                                                                      |
| NoSuchData                 | The requested data could not be found.                                                | 조회된 결과값이 존재하지 않습니다.                      | 404 - Not Found                       | <p>getPurchaseDetails<br>getRecurringPurchaseDetails</p>                                                |
| RequiredValueNotExist      | Request parameters are required. \[ field1, field2, ... ]                             | 필수값이 존재하지 않습니다. \[ field1, field2, ... ] | 400 - Bad Request                     | 공통                                                                                                      |
| ResourceNotFound           | The requested resource could not be found.                                            | 요청한 자원이 존재하지 않습니다.                       | 404 - Not Found                       | 공통                                                                                                      |
| ServiceMaintenance         | System maintenance is in progress.                                                    | 서비스 점검중입니다.                              | 503 - Service Temporarily Unavailable | 공통                                                                                                      |
| Success                    | The request has been completed successfully.                                          | 정상처리 되었습니다.                              | 200 - Success                         | <p>acknowledgePurchase<br>consumePurchase<br>cancelRecurringPurchase<br>reactivateRecurringPurchase</p> |
| UnauthorizedAccess         | Not authorized to this API.                                                           | 해당 API에 접근권한이 없습니다.                      | 403 - Fobidden                        | 공통                                                                                                      |

### 표준오류 응답규격 <a href="#id-06.-api-apiv7" id="id-06.-api-apiv7"></a>

서버 API에서는 정상적인 응답 외에 오류발생 시, 아래의 Example과 같은 형식의 표준오류응답을 리턴합니다. 아래 내용을 참고하시기 바랍니다.

*   **Response Body :** JSON 형식

    | Element Name | Data Type | Data Size | Description |
    | ------------ | --------- | --------- | ----------- |
    | code         | String    | -         | 응답 코드       |
    | message      | String    | -         | 응답 메시지      |
    | error        | Object    | -         | <p><br></p> |
*   **Example**

    ```json
    HTTP/1.1 400 Bad Request
    Content-type: application/json;charset=UTF-8
    {
        "error" : {
            "code" : "NoSuchData",
            "message" : "The requested data could not be found."
        }
    }
    ```

## **공통 코드**  <a href="#id-06.-api-apiv7" id="id-06.-api-apiv7"></a>

### **상품타입 코드**  <a href="#id-06.-api-apiv7" id="id-06.-api-apiv7"></a>

| Code         | Name   | Description     |
| ------------ | ------ | --------------- |
| inapp        | 관리형 상품 | 소비성/영구성/기간제 상품  |
| auto         | 월정액 상품 | 월 자동결제 상품       |
| subscription | 구독형 상품 | 구독형(자동결제) 상품    |
| all          | 전체 상품  | 관리형 상품 + 월정액 상품 |



출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/pns.md
# 07. PNS(Payment Notification Service) 이용하기

## **개요**  

PNS는 Payment Notification Service의 약자입니다. PNS는 모바일의 네트워크 연결 불안정성을 보완하기 위해 개발사가 지정한 서버로 원스토어의 서버가 개별 사용자의 결제 상태(결제 완료, 결제 취소)를 메시지로 전송하여 결제 트랜젝션의 상태를 손실없이 알려주기 위한 용도의 기능입니다. 즉, 개발사가 지정한 서버에서 원스토어가 정의한 규칙에 맞추어 API를 구현하면 해당 API를 원스토어의 결제 담당 서버에서 호출하는 형태입니다.

Server to Server, 즉 서버간에 데이터를 전송한다고 할지라도 네트워크 문제로 메세지 전송 실패가 발생하기 때문에 200 OK로 응답을 인지하지 못할 경우 반복하여 메시지가 전송될 수 있습니다. 개발사의 서버는 메시지를 수신후 정의된 응답을 하여야 원스토어는 개발사 서버가 정상적으로 메시지를 전달 받았음을 인지합니다.

결제 트렌젝션의 유형에 따라 아래와 같은 유형의 메세지가 전송됩니다.

* 인앱상품 결제 또는 결제취소가 발생하면 원스토어가 개발사 서버로 알림을 전송하는 PNS(Payment Notification Service)&#x20;
* 구독 상태가 변경되면 개발사 서버로 알림을 전송하는 SNS (Subscription Notifacation Service)&#x20;

{% hint style="warning" %}
Notification은 발송/수신 서버의 상태에 따라 지연 또는 유실될 수 있으므로, notification 수신을 기준으로 상품(서비스)을 제공하는 것은 권장하지 않습니다.&#x20;

정상적인 결제 건인지 Server to Server로 확인하기를 원하신다면 PNS notification을 이용하는 대신, 관련 서버 API로 조회하는 것을 권장합니다.

원스토어는 검증 및 모니터링 목적으로 결제 테스트를 진행할 수 있으며, 해당 테스트 건들도 결제/결제취소 시 동일하게 notification이 발송됩니다. 원스토어가 진행한 결제 테스트 내역은 주기적으로 원스토어에서 자체 취소 처리합니다.
{% endhint %}

## **PNS 수신 서버 URL 설정** 

PNS를 수신 받을 개발사 서버의 URL은 '개발자센터 > Apps > 상품 선택 > In-App정보' 메뉴에서 'PNS 관리' 버튼을 클릭하면 설정할 수 있습니다.\
URL은 Sandbox(개발용) 결제환경 및 상용(상용테스트 포함) 결제환경을 각각 설정할 수 있으며, 개발용/상용 서버가 동일할 경우 동일한 URL을 입력하시면 됩니다.



## **PNS 상세** 

{% hint style="info" %}
2025년 3월 20일 개발자센터 개편의 영향으로 PNS 3.1.0 버전이 추가 되었습니다.&#x20;

* &#x20;packageName 파라미터가 clientId로 변경 되었으며, 3.0.0 이하 버전에서는 변경 사항이 없습니다.
{% endhint %}

### PNS Payment Notification 메시지 발송 규격 (원스토어 → 개발사 서버) 
* **메세지 이름** : Payment Transaction Notification 
* **설명** : 사용자의 개별 결제건의 상태(결제완료, 취소)포함된 message를 개발사의 서버로 전송(원스토어의 서버가 개발사가 사전에 정의한 url을 post(with json body) 형식으로 호출함)
* **URI** : 개발자 센터에서 설정한 Payment Notification URL
* **Method** : POST
* **Request Parameters** : N/A
*   **Request Header** :&#x20;

    | Parameter Name | Data Type | Description      |
    | -------------- | --------- | ---------------- |
    | Content-Type   | String    | application/json |
* Request Body : JSON 형식

| Element Name       | Data Type     | Description                                                                   |                                       |
| ------------------ | ------------- | ----------------------------------------------------------------------------- | ------------------------------------- |
| msgVersion         | String        | <p>메시지 버전</p><ul><li>개발(Sandbox) : 3.1.0D</li><li>상용(상용테스트) : 3.1.0</li></ul> |                                       |
| clientId           | String        | 앱의 클라이언트 ID                                                                   |                                       |
| productId          | String        | 인앱상품의 상품 ID                                                                   |                                       |
| messageType        | String        | SINGLE\_PAYMENT\_TRANSACTION 고정                                               |                                       |
| purchaseId         | String        | 구매 ID                                                                         |                                       |
| developerPayload   | String        | 구매건을 식별하기 위해 개발사에서 관리하는 식별자                                                   |                                       |
| purchaseTimeMillis | Long          | 원스토어 결제 시스템에서 결제가 완료된 시간(ms)                                                  |                                       |
| purchaseState      | String        | COMPLETED : 결제완료 / CANCELED : 취소                                              |                                       |
| price              | String        | 결제 금액                                                                         |                                       |
| priceCurrencyCode  | String        | 결제 금액 통화코드(KRW, USD, ...)                                                     |                                       |
| productName        | String        | 구매요청 시 개발사가 customized 인앱상품 제목을 설정한 경우 전달                                     |                                       |
| paymentTypeList    | List          | 결제 정보 목록                                                                      |                                       |
| <p><br></p>        | paymentMethod | String                                                                        | 결제 수단 (상세 내용은 아래 paymentMethod 정의 참고) |
| <p><br></p>        | amount        | String                                                                        | 결제 수단 별 금액                            |
| billingKey         | String        | 확장 기능용 결제 키                                                                   |                                       |
| isTestMdn          | Boolean       | 시험폰 여부(true : 시험폰, false : 시험폰 아님)                                            |                                       |
| purchaseToken      | String        | 구매토큰                                                                          |                                       |
| environment        | String        | <p>결제환경</p><ul><li>개발(샌드박스) : SANDBOX</li><li>상용 :COMMERCIAL</li></ul>        |                                       |
| marketCode         | String        | 마켓 구분코드 ( MKT\_ONE : 원스토어, MKT\_STM : 스톰 )                                    |                                       |
| signature          | String        | 본 메시지에 대한 signature                                                           |                                       |

\
**Example**

```
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

\


#### **paymentMethod(원스토어 결제수단) 정의** 

| paymentMethod    | 결제수단 명칭    | 설명                         |
| ---------------- | ---------- | -------------------------- |
| DCB              | 휴대폰결제      | 통신사 요금청구서에 '정보이용료' 항목으로 청구 |
| PHONEBILL        | 휴대폰 소액결제   | 통신사 요금청구서에 '소액결제' 항목으로 청구  |
| ONEPAY           | ONE pay    | 원스토어가 제공하는 간편결제            |
| CREDITCARD       | 신용카드       | 일반 신용카드 결제                 |
| 11PAY            | 11Pay      | 11번가에서 제공하는 신용카드 간편결제      |
| NAVERPAY         | N pay      | 네이버에서 제공하는 네이버페이 결제        |
| CULTURELAND      | 컬쳐캐쉬       | 한국문화진흥에서 제공한는 컬쳐캐쉬 결제      |
| OCB              | OK cashbag | SK플래닛이 제공하는 OK캐쉬백 결제       |
| ONESTORECASH     | 원스토어 캐쉬    | 원스토어 캐쉬 결제                 |
| COUPON           | 원스토어 쿠폰    | 원스토어 쿠폰 결제                 |
| POINT            | 원스토어 포인트   | 원스토어 포인트 결제                |
| TELCOMEMBERSHIP | 통신사멤버십     | 통신사에서 제공하는 멤버십 결제          |
| EWALLET          | e-Wallet   | e-Wallet 결제                |
| BANKACCT         | 계좌결제       | 일반 계좌결제                    |
| PAYPAL           | 페이팔        | 페이팔이 제공하는 결제               |
| MYCARD           | 마이카드       | 소프트월드에서 제공하는 마이카드 결제       |

#### **Signature 검증 방법** 

아래 코드를 사용하면 signature에 대한 위변조 여부를 확인할 수 있습니다.

* 코드 내 PublicKey는 '라이선스 관리'에서 제공되는 라이선스 키를 의미합니다.

{% tabs %}
{% tab title="Java" %}
```
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
{% endtab %}

{% tab title="PHP" %}
```
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
{% endtab %}

{% tab title="Python" %}
{% code overflow="wrap" %}
```
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
{% endcode %}
{% endtab %}
{% endtabs %}

### PNS Subscription Notification 메시지 발송 규격 (원스토어 → 개발사 서버)  
* **메세지 이름** : Subscription Transaction Notification 
* **설명** : 사용자의 개별 정기 결제건의 정기결제 상태정보(subscriptionNotification)가 포함된 message를 개발사의 서버로 전송(원스토어의 서버가 개발사가 사전에 정의한 url을 post(with json body) 형식으로 호출함)
* **URI** : 개발자 센터에서 설정한 Subscription Notification URL
* **Method** : POST
* **Request Parameters** : N/A
*   **Request Header** :&#x20;

    | Parameter Name | Data Type | Description      |
    | -------------- | --------- | ---------------- |
    | Content-Type   | String    | application/json |
* Request Body : JSON 형식

| Element Name             | Data Type        | Description                                                                   |              |
| ------------------------ | ---------------- | ----------------------------------------------------------------------------- | ------------ |
| msgVersion               | String           | <p>메시지 버전</p><ul><li>개발(Sandbox) : 3.1.0D</li><li>상용(상용테스트) : 3.1.0</li></ul> |              |
| clientId                 | String           | 앱의 클라이언트 ID                                                                   |              |
| eventTimeMillis          | Long             | 이벤트 발생 시간                                                                     |              |
| subscriptionNotification | Object           | 결제 정보 목록                                                                      |              |
| <p><br></p>              | version          | String                                                                        | 구독 알람 메시지 버전 |
| <p><br></p>              | notificationType | Integer                                                                       | 구독상태         |
| <p><br></p>              | purchaseToken    | String                                                                        | 구매 토큰        |
| <p><br></p>              | productId        | String                                                                        | 인앱상품의 상품 ID  |
| environment              | String           | <p>결제환경</p><ul><li>개발(샌드박스) : SANDBOX</li><li>상용 :COMMERCIAL</li></ul>        |              |
| marketCode               | String           | 마켓 구분코드 ( MKT\_ONE : 원스토어, MKT\_GLB : 글로벌 원스토어                                |              |

\
**Example**

```
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

\


#### **구독상태 정의** 

| 구독 상태 | 구독 코드                                  | 설명                          |
| ----- | -------------------------------------- | --------------------------- |
| 1     | SUBSCRIPTION\_RECOVERED                | 정기 결제가 보류 상태에서 복구 되었습니다.    |
| 2     | SUBSCRIPTION\_RENEWED                  | 정기 결제가 갱신 되었습니다.            |
| 3     | SUBSCRIPTION\_CANCELED                 | 고객이 정기 결제 해지를 요청 하였습니다.     |
| 4     | SUBSCRIPTION\_PURCHASED                | 새로운 정기 결제 상품이 구매 되었습니다.     |
| 5     | SUBSCRIPTION\_ON\_HOLD                 | 결제 실패로 정기 결제가 보류 상태가 되었습니다. |
| 6     | SUBSCRIPTION\_IN\_GRACE\_PERIOD        | 결제 실패로 정기 결제가 유예 상태가 되었습니다. |
| 7     | SUBSCRIPTION\_RESTARTED                | 고객이 정기 결제 해지 요청을 취소 하였습니다.  |
| 8     | SUBSCRIPTION\_PRICE\_CHANGE\_CONFIRMED | 사용자가 정기 결제 가격 변경에 동의 하였습니다. |
| 9     | SUBSCRIPTION\_DEFERRED                 | 정기 결제 이용 기간이 연장 되었습니다.      |
| 10    | SUBSCRIPTION\_PAUSED                   | 정기 결제가 일시 중지 되었습니다.         |
| 11    | SUBSCRIPTION\_PAUSE\_SCHEDULE\_CHANGED | 정기 결제 일시 중지 일정이 변경 되었습니다.   |
| 12    | SUBSCRIPTION\_REVOKED                  | 정기 결제가 즉시 해지 되었습니다.         |
| 13    | SUBSCRIPTION\_EXPIRED                  | 정기 결제가 만료 되었습니다.            |

\


## Notification 전송 정책 

원스토어의 PNS 서버는 HTTP(S) 요청을 통하여 개발사 서버로 notification을 전송합니다.

이때 개발사 서버는 notification를 정상적으로 수신했다는 의미로 HTTP Status Code를 200으로 응답하여야 합니다.

만약 네트워크 지연으로 인한 유실 또는 개발사 서버의 비정상적인 상황으로 HTTP Status Code를 200으로 응답받지 못한 경우, PNS서버는 notification 전송이 실패했다고 판단하고 3일간 최대 30회의 재전송을 수행하게 됩니다.

notification의 재전송은 아래의 예시와 같이 일정한 delay를 가진 후 실행되며, 재시도 회수가 많아지면 delay가 점차 늘어나는 구조로 되어 있습니다.

**Example**

| 회차     | delay (초) | 재전송 시간              |
| ------ | --------- | ------------------- |
| 0 (최초) | 0         | 2020-05-17 13:10:00 |
| 1      | 30        | 2020-05-17 13:10:30 |
| 2      | 120       | 2020-05-17 13:12:30 |
| 3      | 270       | 2020-05-17 13:17:00 |
| 4      | 480       | 2020-05-17 13:25:00 |
| ...    | ...       | ...                 |



출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/subs.md
# 08. 정기 결제 적용하기

## **개요** <a href="#id-08." id="id-08."></a>

정기 결제는 정해진 기간동안 사용자가 서비스를 이용할 수 있으며, 결제 주기에 따라 정기적으로 결제가 갱신됩니다.&#x20;

일반적으로 음악이나 영화, 혹은 게임들을 이용할 때 이와 같이 정기 결제 서비스를 사용하고 있습니다.  &#x20;

개발자는 다양한 서비스를 제공하기 위해 하나의 앱 안에서 여러가지 정기 결제 상품을 구성할 수 있고, 사용자는 여러가지 정기 결제를 이용할 수 있습니다. &#x20;

또한, 신규 사용자를 확보하기 위해 할인 가격으로 정기 결제를 제공하거나 체험을 위한 무료 기간도 제공할 수 있습니다. &#x20;

마찬가지로 기존 사용자가 이용중인 정기 결제 상품을 변경 할 수 있도록 상품을 전환 시킬 수도 있습니다.&#x20;

## **정기 결제 처리 하기**  <a href="#id-08." id="id-08."></a>

사용자가 정기 결제 상품을 구매하면, 상품을 이용하는 동안 다양한 상태 변경을 거칠 수 있습니다.&#x20;

앱은 정기 결제의 상태를 확인하여, 각 상태 변경에 대응해야 합니다.&#x20;

정기 결제의 상태는 원스토어 인앱 결제 라이브러리의 PurchaseClient.queryPurchasesAsync() 또는 서버 API getSubscriptionDetail을 사용하여 확인 할 수 있습니다.&#x20;

|      | PurchaseClient.queryPurchasesAsync() |                | getSubscriptionDetail |                                     |              |
| ---- | ------------------------------------ | -------------- | --------------------- | ----------------------------------- | ------------ |
| 상태   | 반환 여부                                | recurringState | 반환 여부                 | expiryTimeMillis                    | autoRenewing |
| 구독중  | 예                                    | 0              | 예                     | 향후 일정                               | true         |
| 해지   | 예                                    | 1              | 예                     | 향후 일정                               | false        |
| 유예   | 예                                    | 0              | 예                     | 향후 일정(유예 기간 종료)                     | true         |
| 보류   | 아니오                                  | 0              | 예                     | 종료됨(예상 만료 시간 종료 또는 유예 기간 종료(있는 경우)) | true         |
| 일시중지 | 아니오                                  | 0              | 예                     | 종료됨                                 | true         |
| 만료   | 아니오                                  | 1              | 예                     | 종료됨                                 | false        |

정기 결제 상태가 변경되면, SubscriptionNotification이 전송됩니다. 관련 내용은 PNS(Payment Notification Service) 이용하기를 참고하세요.

### 정기 결제 구매  <a href="#id-08." id="id-08."></a>

사용자가 정기 결제를 구매하면 PurchaseClient.queryPurchasesAsync()에서 정기 결제 결과를 반환하고, SUBSCRIPTION\_PURCHASED인 SubscriptionNotification이 전송됩니다.&#x20;

이 알림을 받으면 원스토어 인앱결제 API를 쿼리하여 정기 결제 상태를 업데이트 해야 합니다. 정기 결제 리소스는 다음 예시를 참고하세요.&#x20;

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

### 정기 결제 갱신 <a href="#id-08." id="id-08."></a>

정기 결제가 갱신되면 SUBSCRIPTION\_RENEWED 알림이 전송됩니다.&#x20;

이 알림을 받으면 원스토어 인앱결제 API를 쿼리하여 새로운 만료 일시로 정기 결제 상태를 업데이트 해야 합니다.  정기 결제 리소스는 다음 예시를 참고하세요.&#x20;

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

월간 상품의 경우 익월 동일일에 결제가 됩니다. (3개월, 6개월 상품도 동일하게 동작합니다.)&#x20;

예를들어, 1월 15일에 정기 결제가 시작되었다면, 2월 15일에 다음 결제가 발생합니다.&#x20;

하지만 동일일이 없는 경우에는 그 달의 마지막 날이 결제 갱신일이 됩니다. &#x20;

예를들어, 1월 31일에 정기 결제가 시작되었다면, 2월 28일(혹은 29일)에 다음 결제가 발생하고, 3월 28일(혹은 29)에 그 다음 결제가 발생합니다.&#x20;

### 정기 결제 만료 <a href="#id-08." id="id-08."></a>

정기 결제가 만료되면 PurchaseClient.queryPurchasesAsync()에서 더 이상 반환되지 않으며 사용자는 정기 결제를 이용할 수 없게 됩니다.&#x20;

SUBSCRIPTION\_EXPIRED 인 SubscriptionNotification도 정기 결제가 만료될 때 전송됩니다.&#x20;

이 알림을 받으면 원스토어 인앱결제 API를 쿼리하여 정기 결제 상태를 업데이트 해야 합니다. 정기 결제 리소스는 다음 예시를 참고하세요.&#x20;

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

### 정기 결제 해지  <a href="#id-08." id="id-08."></a>

정기 결제는 사용자가 원스토어 앱에서 정기 결제를 직접 해지할 수 있고, 결제 수단에 문제가 생겼을 때 이를 해결하지 않는 경우 자동으로 해지 될 수 있습니다.   &#x20;

원스토어 인앱결제 API cancelRecurringPruchase 를 사용하여 정기 결제를 해지할 수도 있습니다.&#x20;

사용자가 정기 결제를 직접 해지 하더라도 현재 정기 결제 기간이 끝날 때까지는 서비스를 이용할 수 있어야 합니다. &#x20;

정기 결제가 해지 되었지만 아직 만료되지 않은 경우 PurchaseClient.queryPurchasesAsync() 에서 반환됩니다. 정기 결제가 해지되면 SUBSCRIPTION\_CANCELED 알림이 전송됩니다.&#x20;

이 알림을 받으면 원스토어 인앱결제 API를 쿼리하여 상태를 업데이트 해야 합니다. 쿼리 시 autoRenewing=false와 사용자가 서비스를 이용 할 수 있는 기한 expiryTimeMillis 이 반환 됩니다.&#x20;

expiryTimeMillis가 기한이 지난 과거라면 즉시 서비스가 중단되고, 기한이 미래라면 해당일시까지 서비스를 이용 할 수 있습니다. 정기 결제 리소스는 다음 예시를 참고하세요.    &#x20;

```reason
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

사용자에게 정기 결제가 해지 되었으며, 해당 정기 결제의 만료일시를 앱을 통해 알려야합니다. (예시 : 정기결제가 해지되었습니다. 2023년 6월 13일에 정기 결제가 만료됩니다. \[확인]) &#x20;

### 정기 결제 취소 <a href="#id-08." id="id-08."></a>

다양한 이유로 사용자가 정기 결제를 취소 할 수 있습니다. 정기 결제가 취소되면 즉시 서비스 이용이 차단되어야 합니다.&#x20;

이 경우 `PurchaseClient.queryPurchasesAsync()`에서 더이상 반환되지 않으며, SUBSCRIPTION\_REVOKED 알림도 전송 됩니다.&#x20;

이 알림을 받으면 원스토어 인앱결제 API 를 쿼리하여 정기 결제 상태를 업데이트 해야 합니다.&#x20;

정기 결제 리소스에는 autoRenewing와 정기 결제 이용 기한인 expiryTimeMillis가 포함됩니다. 정기 결제 리소스는 다음 예시를 참고하세요.&#x20;

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

### 정기 결제 유예  <a href="#id-08." id="id-08."></a>

사용자가 정기 결제 이용 중 결제 수단에 문제가 발생하고 문제가 해결되지 않는 경우 정기 결제가 바로 해지 되지는 않습니다.&#x20;

위와 같은 경우 (1) 결제 유예 - 유예 기간이 설정된 경우  (2) 계정 보류 순서대로 정기 결제 상태가 변경됩니다. &#x20;

결제 유예 기간 제공여부는 원스토어 개발자센터에서 설정 할 수 있습니다. &#x20;

유예 기간을 설정 했다면 사용자는 해당 기간 동안 정기 결제 콘텐츠를 이용할 수 있어야 합니다.&#x20;

앱에서 PurchaseClient.queryPurchasesAsync() 를 사용하여 정기 결제 상태를 확인한다면, PurchaseClient.queryPurchasesAsync()가 만료일 전에 정기 결제 상태를 계속 반환하므로 앱은 유예 기간을 자동으로 처리해야 합니다. &#x20;

사용자의 정기 결제 상태가 결제 유예가 되는 경우 SUBSCRIPTION\_IN\_GRACE\_PERIOD 인 알림이 전송됩니다.

이 알림을 받으면 원스토어 인앱결제 API를 쿼리하여 상태를 업데이트 해야합니다. 정기 결제 리소스에는 autoRenewing=true 와 유예가 적용되도록 미래 시점의 expiryTimeMillis 이 포함됩니다. \


유예 기간 내에 결제 문제를 해결하지 않으면 계정 보류 상태가 되어 사용자가 정기 결제 콘텐츠를 이용 할 수 없습니다.&#x20;

앱에서는 메시지를 표시하여 사용자가 결제 문제를 해결 할 수 있도록 해야 합니다. &#x20;

사용자가 결제 문제를 해결하는데 도움이 되도록 딥링크를 제공할 수 있습니다. &#x20;

사용자가 결제 문제를 해결하면 즉시 정기 결제가 갱신 됩니다.&#x20;

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

### 정기 결제 계정 보류 <a href="#id-08." id="id-08."></a>

사용자가 정기 결제 이용 중 결제 수단에 문제가 발생하고 문제가 해결 되지 않는 경우 유예 기간이 종료되면 계정 보류 상태가 됩니다.&#x20;

계정 보류 기간은 최대 30일이며, 유예 기간과는 다르게 계정 보류 상태에서는 정기 결제 콘텐츠를 이용 할 수 없습니다.

또한 계정 보류 중에는 정기 결제가 PurchaseClient.queryPurchasesAsync()에 의해 반환되지 않습니다.&#x20;

사용자의 정기 결제가 계정 보류 상태가 되면  SUBSCRIPTION\_ON\_HOLD 알림이 전송됩니다.&#x20;

이 알림을 받으면 원스토어 인앱결제 API를 호출하여 정기 결제 정보를 업데이트 해야합니다. 계정 보류 상태에서는 정기 결제 리소스의 expiryTimeMillis가 과거로 설정 됩니다. \


보류 상태에서는 사용자가 정기 결제 콘텐츠를 이용 할 수 없고, 지정된 보류 기간이 지나면 정기 결제가 해지됩니다.&#x20;

따라서 앱에서는 메시지를 표시하여 사용자가 문제를 해결 할 수 있도록 해야합니다.&#x20;

사용자가 문제를 해결하는데 도움이 되도록 딥링크를 제공할 수 있습니다. &#x20;

사용자가 문제를 해결하여 정기 결제가 복구되면 정기 결제일은 복구된 날짜로 변경됩니다.   &#x20;

계정 보류 기간이 종료되기 전에 사용자가 문제를 해결하지 않으면 SUBSCRIPTION\_CANCELED 알림이 전송 되며, 원스토어 인앱결제 API를 쿼리하여 정기 결제 상태를 업데이트 해야합니다.&#x20;

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

### 정기 결제 일시 중지 <a href="#id-08." id="id-08."></a>

개발자는 원스토어 개발자센터에서 일시 중지 기능을 사용하도록 설정할 수 있습니다.&#x20;

일시 중지 기능을 사용하도록 설정하면, 사용자는 앱의 구독을 취소하는 대신 일정 기간동안 정기 결제를 일시 중지 할 수 있습니다.&#x20;

일시 중지는 현재 정기 결제 중인 상품의 기한이 만료된 후 시작됩니다. 일시중지 된 동안에는 사용자가 정기 결제 콘텐츠를 이용할 수 없어야 합니다.&#x20;

일시중지 기간이 끝나면 정기 결제가 다시 시작됩니다.&#x20;

사용자는 일시중지 기간이 끝나기 전에 직접 일시중지를 해제하여 정기 결제를 다시 시작할 수도 있습니다. 이 경우 일시중지를 해제하여 정기 결제를 다시 시작한 날짜가 정기 결제일이 됩니다.&#x20;

일시중지 중에는 PurchaseClient.queryPurchasesAsync()에 의해 반환되지 않습니다.  정기 결제가 재개되면 PurchaseClient.queryPurchasesAsync()에 의해 반환됩니다.&#x20;

사용자가 일시중지를 선택하면 SUBSCRIPTION\_PAUSE\_SCHEDULE\_CHANGED인 SubscriptionNotification이 전송됩니다. &#x20;

이 시점에 사용자는 정기 결제 이용기간이 남아있으며, 따라서 정기 결제 콘텐츠를 계속 이용할 수 있어야 합니다. &#x20;

정기 결제 리소스에는 autoRenewing=true, paymentState=1과 미래의 expiryTimeMillis, autoResumeTimeMillis 값이 포함됩니다.&#x20;

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



일시중지가 시작되면 SUBSCRIPTION\_PAUSED 인 Subscription Notification이 전송됩니다.&#x20;

일시중지가 시작되면 정기 결제 이용기간이 만료 된 것이며, 따라서 정기 결제 콘텐츠는 이용할 수 없어야 합니다.&#x20;

정기 결제 리소스에는 autoRenewing=true, paymentState=0과 과거의 expiryTimeMillis, autoResumeTimeMillis 값이 포함됩니다.&#x20;

일시중지 기간이 끝나서 정기 결제가 자동으로 재개되거나 사용자가 일시중지를 해제하여 정기 결제를 재개하면 SUBSCRIPTION\_RENEWED 인 SubscriptionNotification이 전송됩니다.&#x20;

이 경우 [정기 결제 갱신](#id-08.-3)과 동일한 처리가 필요합니다.&#x20;

정기 결제가 재개될 때 결제에 문제가 생기면 [정기 결제 계정 보류](#id-08.-8) 상태가 발생하며, [정기 결제 계정 보류](#id-08.-8) 상태와 동일하게 처리되어야 합니다.&#x20;

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

### 상품 변경(업그레이드, 다운그레이드) <a href="#id-08." id="id-08."></a>

사용자가 정기 결제 중인 상품을 다른 정기 결제 상품으로 변경하면 기존에 사용 중인 정기 결제는 무효화되고, 새로운 정기 결제가 새 구매 토큰과 함께 생성됩니다.&#x20;

또한 정기 결제 리소스에는 기존 정기 결제를 나타내는 linkedPurchaseToken 이 포함됩니다.

linkedPurchaseToken 을 사용하면 이전 정기 결제를 조회하여 새로운 정기 결제를 동일한 계정과 연결할 수 있습니다.&#x20;

새로운 정기 결제가 생성 되었으므로 해당 결제 건에 대한 [구매 확인](sdk)이 필요합니다. 정기 결제 리소스는 다음 예시를 참고하세요. &#x20;

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

### **정기 결제 관리 화면 열기**    <a href="#id-08." id="id-08."></a>

원스토어는 사용자가 정기 결제 상품을 쉽게 관리하도록 정기 결제 관리 메뉴를 제공합니다. &#x20;

매개변수로 _SubscriptionsParam&#x73;_&#xC5D0; _PurchaseDat&#x61;_&#xB97C; 포함해서 넣으면 구매 데이터를 확인하여 해당 정기 결제 상품의 관리 화면을 실행합니다.\
&#xNAN;_&#x53;ubscriptionParam&#x73;_&#xC744; null 넣을 경우 사용자의 정기 결제 리스트 화면을 실행합니다.

다음은 정기 결제 관리 화면을 띄우는 방법을 나타내는 예제입니다.

{% tabs %}
{% tab title="kotlin" %}
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
{% endtab %}

{% tab title="java" %}
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
{% endtab %}
{% endtabs %}

| 규격 Uri       | onestore://common/subscription/payment/{caller\_package}?purchase\_token={purchase\_token}                                      |                            |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| 규격 parameter | <p>caller_package</p><p>(mandatory)</p>                                                                                         | 연동 규격을 호출 할 앱의 packageName |
|              | <p>purchase_token</p><p>(optional)</p>                                                                                          | 특정 구매 상세로 가기위한 구매 Token    |
| 상세 정보        | <ul><li>마이페이지 > 정기결제 목록 화면으로 이동한다.</li><li>purchase_token 이 존재하는 경우 해당 구매 Token의 정기결제 상세로 이동한다. </li></ul>                      |                            |
| 지원버전         | <p>ONE store Client v7.9.0 이상 ( android:versionName=”7.9.0” android:versionCode=”70900” )</p><p>ONE store Service v7.140 이상</p> |                            |

### **정기 결제 상품 변경 하기**  <a href="#id-08." id="id-08."></a>

사용자는 정기 결제 이용 중에 좀 더 좋은 혹은 저렴한 정기 결제 상품으로 플랜을 변경하고 싶을 수 있습니다.&#x20;

사용자는 다른 정기 결제 상품을 결제하여 정기 결제 상품을 변경 할 수 있고, 개발자는 비례 배분 모드를 설정하여 정기 결제 상품 변경을 처리 할 수 있습니다.&#x20;

설정 할 수 있는 비례 배분 모드(_PurchaseFlowParams.ProrationMode)_&#xB294; 다음과 같습니다.&#x20;

| 비례 배분 모드                                | 설명                                                                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| IMMEDIATE\_WITH\_TIME\_PRORATION        | 정기 결제 상품이 즉시 변경 되며, 가격 차이를 기반으로 정기 결제 갱신일이 조정됩니다.                                                                         |
| IMMEDIATE\_AND\_CHARGE\_PRORATED\_PRICE | <p>정기 결제 상품이 즉시 변경 되며, 정기 결제 갱신일은 동일하게 유지됩니다. 나머지 기간에 대한 가격 차이를 기반으로 요금이 청구됩니다.</p><p>(이 모드는 상품 업그레이드 시에만 적용 가능합니다.) </p> |
| IMMEDIATE\_WITHOUT\_PRORATION           | 정기 결제 상품이 즉시 변경 되며, 다음 결제일에 새로운 가격이 청구됩니다. 정기 결제 갱신일은 동일하게 유지됩니다.                                                         |
| DEFERRED                                | 기존 정기 결제 상품이 만료되면 교체가 적용되며 새로운 정기 결제 상품 요금이 청구됩니다.                                                                        |

비례 배분 예

사용자는 현재 정기 결제 상품 A를 매월 정기 결제하고 있습니다. 이 정기 결제는 2,000원 의 요금이 청구되며 매월 1일에 갱신 됩니다.

4월 15일에 사용자는 요금이 **연간 36,000원** 인 연간 정기 결제 상품 B로 상품을 변경 하기로 했습니다.\


#### **IMMEDIATE\_WITH\_TIME\_PRORATION**

정기 결제 상품 A가 즉시 종료됩니다. 사용자는 한 달(4월 1일\~30일) 요금을 결제 했는데 정기 결제 기간 중간에 상품을 변경 했으므로 월간 정기 결제 요금의 절반(1,000원)은 새 정기 결제에 적용됩니다. 그러나 새 정기 결제 요금은 연간 36,000원이므로 1,000원의 잔액은 10일(4월 16일\~25일)에 해당합니다. 따라서 4월 26일에 새 정기 결제의 요금으로 36,000원이 청구되며, 다음 해부터 매년 4월 26일에 36,000원이 청구됩니다.&#x20;

#### **IMMEDIATE\_AND\_CHARGE\_PRORATED\_PRICE**

시간 단위 당 정기 결제 상품 B의 가격(36,000원/연 = 3,000원/월)이 정기 결제 상품 A의 가격(2,000원/월)보다 높으므로 이 모드를 사용할 수 있습니다.  정기 결제 상품A는 즉시 종료됩니다. 사용자는 한 달 전체 요금을 결제 했는데 기간의 절반만 사용했으므로 월간 정기 결제 요금의 절반(1,000원)은 새 정기 결제에 적용됩니다. 그러나 새 정기 결제 요금은 1년에 36,000원 이므로 나머지 15일의 요금은 1,500원입니다. 따라서 새 정기 결제 요금으로 500원의 차액이 청구됩니다. 5월 1일에는 새 정기 결제 상품 B의 요금으로 36,000원이 청구되며, 다음 해부터 매년 5월 1일에 36,000원이 청구됩니다.&#x20;

#### **IMMEDIATE\_WITHOUT\_PRORATION**

정기 결제 상품A가 추가 비용 없이 정기 결제 상품B로 즉시 변경 됩니다. 그리고 5월 1일에 새 정기 결제 상품 B의 요금으로 36,000원이 청구되며, 다음 해부터 매년 5월 1일에 36,000원이 청구됩니다.

#### **DEFERRED**

정기 결제 상품A는 4월 30일에 만료될 때까지 계속됩니다. 5월 1일에 정기 결제 상품 B가 적용되며, 새 정기 결제 등급 요금으로 사용자에게 36,000원이 청구됩니다.



정기 결제는 "_구매 요청하기_"와 동일한 API를 사용하여 사용자에게 상품 변경 기능을 제공할 수 있습니다. 다만, 정기 결제의 상품 변경을 위해선 기존 정기 결제 구매 토큰과 비례 배분 모드 값이 필수로 필요합니다.

다음 예와 같이 현재 정기 결제와 변경하여 적용 될 정기 결제 및 비례 배분 모드에 관한 정보를 제공해야 합니다.

{% tabs %}
{% tab title="Kotlin" %}
```kotlin
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
{% endtab %}

{% tab title="Java" %}
```java
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
{% endtab %}
{% endtabs %}

즉시 변경이 이루어지는 비례 배분 모드의 경우 [구매 요청하기](sdk) 로직을 수행하기 때문에 응답은 _PurchasesUpdatedListene&#x72;_&#xC5D0;서 수신합니다. 또한 [구매내역 조회하기](sdk)에서도 요청 시 응답을 받을 수 있습니다.

&#x20;_PurchaseClient.acknowledgeAsync()또는, acknowledgePurchase를_ 사용하여 [구매 확인](sdk)처리를 해야 합니다.

원스토어 인앱결제 API는 정기 결제 리소스에서 linkedPurchaseToken 을 반환합니다. linkedPurchaseToken 을 무효화하여 콘텐츠 이용 권한을 얻는 데 이전 토큰이 사용되지 않도록 해야 합니다.&#x20;

지연 변경이 이루어지는 비례 배분 모드의 경우 정기 결제 구매 및 변경 여부와 함께 _PurchasesUpdatedListener_  호출을 수신합니다.&#x20;

변경이 적용될 때까지 PurchaseClient.queryPurchasesAsync() 는 기존 정기 결제 상품 정보의 구매 정보를 계속 반환합니다.

변경이 적용되면 PurchaseClient.queryPurchasesAsync() 는 새로운 정기 결제 상품의 구매 정보를 반환하고, SUBSCRIPTION\_RENEWED 인 Subscription Notification 이 전송됩니다.

지연 변경이 이루어지는 경우 이 알림을 수신하여 _acknowledgePurchase_ 을 사용하여 구매확인 처리를 하는 것이 좋습니다.&#x20;



**프로모션이 있는 상품으로 변경하기**&#x20;

원스토어에서는 프로모션이 적용되어 있는 정기 결제 상품을 이용 중인 경우 다른 상품으로 변경 할 수 없습니다.&#x20;

다만, 변경 대상이 되는 상품은 프로모션이 적용되어 있어도 변경이 가능합니다.&#x20;

변경 대상이 되는 상품의 프로모션을 고객이 계속 이용하려면 IMMEDIATE\_WITH\_TIME\_PRORATION 모드를 적용해야 합니다.&#x20;

그 외 다른 비례 배분 모드의 경우 변경 대상이 되는 상품에 프로모션이 적용 되어 있더라도, 프로모션을 적용 받을 수 없습니다.&#x20;

### **정기 결제 기한 연기하기**   <a href="#id-08." id="id-08."></a>

원스토어 인앱결제 API의 deferSubscription 을 사용하여 정기 결제 사용자의 다음 결제일을 연기할 수 있습니다.&#x20;

연기 기간 동안 사용자는 콘텐츠에 엑세스 할 수 있습니다.&#x20;

정기 결제 기한 연기는 다음과 같은 경우 사용 할 수 있습니다.&#x20;

* 특별 이벤트를 통해 기존 사용자에게 일주일 동안 추가로 정기 결제를 사용할 수 있게 해줍니다.
* 시스템 장애 등의 사유로 고객에게 추가 이용 기간을 부여할 수 있습니다.&#x20;

API 호출 당 최소 하루에서 최대 1년까지 결제 기한을 연기 할 수 있습니다. 결제를 더 연기하려면 기한 내에 다시 API를 호출하면 됩니다. \


결제 연기 시 이메일 혹은 앱 내에서 알림을 보내 결제일이 변경 되었음을 알릴 수 있습니다.&#x20;

### **정기 결제 가격 변경하기** <a href="#id-08." id="id-08."></a>

원스토어 개발자센터에서 정기 결제 상품의 가격을 변경 할 수 있습니다.  &#x20;

정기 결제 상품의 기본 가격을 변경하면, 새로운 구매는 즉시 변경된 가격으로 반영됩니다.&#x20;

다만, 현재 정기 결제를 사용중인 고객의 경우 7일 후에 변경된 가격에 대한 알림이 전달 되며, 이후 30일 간 가격 변경에 대한 동의 기간이 제공됩니다.&#x20;

{% hint style="info" %}
단, Deferred 모드로 변경 예정인 구독 상품의 가격이 변경 된 경우에는

상품이 변경 된 후 가격 변경 동의 흐름이 시작 됩니다.
{% endhint %}

기간 내에 가격 변경에 대한 동의가 이루어지지 않으면, 해당 정기 결제는 다음 정기 결제일에 취소 됩니다.&#x20;

(가격이 내려가거나 유지되는 경우 별도의 알림이나 동의 절차 없이 7일 이후 다음 갱신일에 변경된 가격으로 결제 됩니다.)

#### 가격 변경에 대한 동의하는 경우   <a href="#id-08." id="id-08."></a>

사용자가 구독료 인상에 동의하는 경우 또는 가격을 인하한 경우에는 동의 기간 이후 변경된 가격으로 결제됩니다.&#x20;

또한 SUBSCRIPTION\_PRICE\_CHANGE\_CONFIRMED 인 Subscription Notification이 전송됩니다.&#x20;

#### 가격 변경에 대해 동의하지 않는 경우  <a href="#id-08." id="id-08."></a>

사용자가 구독료 인상에 동의하지 않는 경우 자동으로 정기 결제가 취소되고, SUBSCRIPTION\_EXPIRED 인 Subscription Notification이 전송됩니다.&#x20;

#### 실수로 가격 변경을 변경한 경우  <a href="#id-08." id="id-08."></a>

실수로 정기 결제 가격을 변경 했다면 가격을 다시 되돌리는 것이 좋습니다.&#x20;

7일 이내에 가격을 되돌린다면, 기존 정기 결제 사용자에게 영향을 미치지 않습니다.

다만, 이 경우에도 신규 구매 고객은 변경 된 가격으로 결제하게 되므로 가격 변경은 신중해야 합니다.&#x20;

#### 두 번의 가격 변경이 발생한 경우  <a href="#id-08." id="id-08."></a>

첫 번째 가격 변경 이후 두 번째 가격 변경이 7일 이내라면, 첫 번째 가격 변경은 무효화되고 두 번째 가격 변경만 적용이 됩니다.&#x20;

다만, 7일 이후라면 사용자는 두 번의 가격 변경에 대한 동의를 해야합니다.&#x20;

가격 변경에 대한 동의 절차는 사용자의 불편을 초래하며, 해당 정기 결제 상품을 유지하는 고객이 줄어들 가능성이 높아집니다.&#x20;

다시 한 번 강조하지만 가격 변경은 신중해야 합니다.&#x20;



출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/releasenote.md
# 09. 원스토어 인앱결제 릴리즈 노트

### 원스토어 인앱결제 라이브러리 21.02.00 업데이트

IAP SDK 21.02.00 부터 아래와 같이 `onestore:dev_option`의 `android:value` 값을 설정하면, SDK와 연동되는 스토어 앱을 지정 할 수 있습니다.

```xml
<manifest>
    <application>
        <meta-data android:name="onestore:dev_option" android:value="onestore_01" />
    </application>
</manifest>
```

<table><thead><tr><th width="226">값 (android:value)</th><th>대상 국가 및 지역</th></tr></thead><tbody><tr><td><code>onestore_00</code></td><td>대한민국 (South Korea) <em>(기본값)</em></td></tr><tr><td><code>onestore_01</code></td><td>싱가포르, 타이완 (Singapore, Taiwan)</td></tr><tr><td><code>onestore_02</code></td><td>미국 – Digital Turbine (United States)</td></tr></tbody></table>

StoreEnvironment API 기능 추가

`StoreEnvironment.getStoreType()` API는 SDK가 탑재된 애플리케이션이 원스토어를 통해 설치되었는지를 판단하는 기능을 제공합니다.

Store Type 정의

해당 API는 `StoreType`을 반환하며, 아래 네 가지 값 중 하나를 가집니다.

<table><thead><tr><th width="240">StoreType</th><th width="71">value</th><th>description</th></tr></thead><tbody><tr><td><code>StoreType.UNKNOWN</code></td><td>0</td><td>앱 설치 스토어 정보를 알 수 없음 <em>(APK 직접 설치, 출처 불명 등)</em></td></tr><tr><td><code>StoreType.ONESTORE</code></td><td>1</td><td>ONE Store에서 설치됨 <em>(또는 개발자 옵션이 활성화된 경우)</em></td></tr><tr><td><code>StoreType.VENDING</code></td><td>2</td><td>Google Play Store에서 설치됨</td></tr><tr><td><code>StoreType.ETC</code></td><td>3</td><td>기타 스토어에서 설치됨</td></tr></tbody></table>

### **원스토어 인앱결제 라이브러리 SDK V21.01 배포** <a href="#id-09.-apiv7-sdkv21" id="id-09.-apiv7-sdkv21"></a>

원스토어 인앱결제 SDK V21.01이 배포 되었습니다.&#x20;

주요 변경 사항은 다음과 같습니다.&#x20;

* 앱 무결성을 체크하는 보안 솔루션과 충돌이 있는 문제가 해결 되었습니다.&#x20;
* 국내 원스토어 앱이 설치되어 있을 때, 글로벌 원스토어 앱 다운로드 설치 동선에서 발생하던 로딩 문제가 해결 되었습니다.&#x20;
* 개발 단계에서 개발 환경을 변경 할 수 있는 설정을 제공합니다.



### **원스토어 인앱결제 라이브러리 API V7(SDK V21) 출시** <a href="#id-09.-apiv7-sdkv21" id="id-09.-apiv7-sdkv21"></a>

원스토어 인앱결제 라이브러리 API V7(SDK V21.00)가 출시 되었습니다.  주요한 변경 사항은 다음과 같습니다.&#x20;

새로운 상품은 원스토어 인앱결제 라이브러리 API V7(SDK V21)를 적용해야 합니다.

원스토어 인앱결제 API V5(SDK V17)이상을 적용한 앱은 API V7(SDK V21)로 마이그레이션이 가능합니다.

원스토어 인앱결제 API V4(SDK V16)은 이후 버전과 전혀 다른 구조로 설계 되었기 때문에, 원스토어 인앱결제 API V6(SDK V19)로 마이그레이션 할 수 없습니다.

#### &#x20;구독형 상품  <a href="#id-09." id="id-09."></a>

SDK V19 이하 버전에서 사용하던 월정액 상품을 대신하여 구독형 상품을 제공합니다.&#x20;

구독형 상품은 사용자가 원스토어 앱 내에서 정기 결제 상품을 직접 관리 할 수 있는 다양한 기능을 제공하고&#x20;

신규 유저 유입을 위한 프로모션, 정기 결제 상태 변경에 대한 알림 기능 등 개발자를 위한 다양한 기능도 제공됩니다. &#x20;

\


#### 로그인 프로세스 변경  <a href="#id-09." id="id-09."></a>

기존에 사용되든 PurchaseClient.launchLoginFlowAsync()는 지원되지 않을 예정입니다.

신규로 만들어진 GaaSignInClient를 이용하여 로그인이 지원됩니다. 클래스명이 바뀌었지만 사용방식은 동일합니다.

\


#### &#x20;배포 방식 변경 <a href="#id-09." id="id-09."></a>

기존에는 guthub을 통한 배포였으나, V21부터는 maven을 통해 SDK가 배포 됩니다.&#x20;

이제부터는 SDK의 업데이트가 훨씬 간편 해졌습니다.&#x20;

### **원스토어 인앱결제 라이브러리 API V6(SDK V19) 출시** <a href="#id-09.-apiv6-sdkv19" id="id-09.-apiv6-sdkv19"></a>

원스토어 인앱결제 라이브러리 API V6(SDK V19.00)가 출시 되었습니다.  주요한 변경 사항은 다음과 같습니다.&#x20;

새로운 상품은 원스토어 인앱결제 라이브러리 API V6(SDK V19)를 적용해야 합니다.

원스토어 인앱결제 API V5(SDK V17)을 적용한 앱은 API V6(SDK V19)로 마이그레이션이 가능합니다.

원스토어 인앱결제 API V4(SDK V16)은 이후 버전과 전혀 다른 구조로 설계 되었기 때문에, 원스토어 인앱결제 API V6(SDK V19)로 마이그레이션 할 수 없습니다.

#### &#x20;원스토어 인앱결제 API 버전 삭제  <a href="#id-09.-api" id="id-09.-api"></a>

이제 앱의 매니페스트에서 인앱 API 버전을 추가할 필요가 없습니다.

인앱결제 SDK V19에 API Version meta-data가 추가됩니다.

#### 구매 확인  <a href="#id-09." id="id-09."></a>

원스토어 인앱결제 라이브러리 API V6(SDK V19) 이상을 사용하는 앱에서는 3일 이내에, 구매 확인을 해야합니다.&#x20;

3일 이내에 구매 확인이 되지 않으면, 아이템이 정상적으로 지급이 되지 않은 것으로 판단하고 구매를 취소 합니다.&#x20;

다음 메서드 중 하나를 사용하여 구매를 확인할 수 있습니다.

* 소모성 제품인 경우 PurchaseClient.consumeAsync()를 사용합니다.
* 소모성 제품이 아니라면 PurchaseClient.acknowledgeAsync()를 사용합니다.&#x20;

월정액 상품의 경우에는 최초의 결제에 대해서만 구매를 확인하면 됩니다. &#x20;

원스토어 인앱결제 API V6(SDK V19)를 적용한 경우, 반드시 3일 이내에 구매 확인을 해야합니다.

3일 이내에 구매 확인이 되지 않은 경우 구매가 취소되니 주의해야 합니다.

\


#### 마켓 구분 코드 얻기 <a href="#id-09." id="id-09."></a>

IAP 라이브러리 V6부터 Server to Server API를 사용하기 위해서는 마켓 구분 코드가 필요합니다.

getStoreInfoAsync()를 통해서 마켓 구분 코드를 획득 할 수 있습니다.&#x20;

#### PurchaseClient API 변경  <a href="#id-09.-purchaseclientapi" id="id-09.-purchaseclientapi"></a>

| <p><br></p>                 | V5 (SDK V17)                | V6(SDK V19)                 |
| --------------------------- | --------------------------- | --------------------------- |
| 결제모듈과의 연결                   | connect                     | startConnection             |
| 결제모듈과의 연결해제                 | terminate                   | endConnection               |
| 지원여부 확인                     | isBillingSupportedAsync     | X                           |
| 인앱 상품 구매                    | launchPurchaseFlowAsync     | launchPurchaseFlow          |
| 인앱 상품 소비                    | consumeAsync                | consumeAsync                |
| 인앱 상품 구매 확인                 | X                           | acknowledgeAsync            |
| 소비되지 않은 상품의 구매내역(월 자동결제 포함) | queryPurchasesAsync         | queryPurchasesAsync         |
| 인앱 상품 상세정보                  | queryProductsAsync          | queryProductDetailsAsync    |
| 월 자동결제 상태 변경                | manageRecurringProductAsync | manageRecurringProductAsync |
| 결제 모듈 업데이트 또는 설치            | launchUpdateOrInstallFlow   | launchUpdateOrInstallFlow   |
| 원스토어 로그인 호출                 | launchLoginFlowAsync        | launchLoginFlowAsync        |
| 마켓 구분 코드 확인                 | X                           | getStoreInfoAsync           |

#### &#x20;Server API 변경  <a href="#id-09.-serverapi" id="id-09.-serverapi"></a>



| 구분           | API 목록              | API URI - V5(SDK V17)                                                               | API URI - V6(SDK V19)                                                                            | 변경사항                                                                                                                      |
| ------------ | ------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| V5 (SDK V17) | V6 (SDK V19)        |                                                                                     |                                                                                                  |                                                                                                                           |
| 변경           | OAuth 토큰발급          | /v2/oauth/token                                                                     | /v6/oauth/token                                                                                  | <p>응답규격 중 status 필드 삭제</p><p>Http response code로 정상(200) 여부 확인</p>                                                        |
| 삭제           | 구매상품 상세조회           | /v2/purchase/details/{purchaseId}/{packageName}                                     | N/A                                                                                              | <p>보안 강화를 위하여 구매상품 상세조회 API는 삭제</p><p>구매상품 상세조회 By 인앱상품ID로 통합</p>                                                         |
| 변경           | 구매상품 상세조회 By 인앱상품ID | <p>/v2/purchase/details-by-productid<br>/{purchaseId}/{packageName}/{productId}</p> | <p>/v6/apps/{packageName}/purchases/inapp/products<br>/{productId}/{purchaseToken}</p>           | 응답 규격에 purchaseId, acknowledgeState 필드 추가                                                                                 |
| 변경           | 월정액 상품 구매 상세조회      | /v2/purchase/recurring-details/{purchaseId}/{packageName}                           | <p>/v6/apps/{packageName}/purchases/auto/products<br>/{productId}/{purchaseToken}</p>            | <p>응답규격에 acknowledgeState, lastPurchaseId, lastPurchaseState 추가</p><p>응답규격에 price, developerPayload, purchaseState 삭제</p> |
| 삭제           | 월정액 상품 마지막 구매 상세조회  | /v2/purchase/last-recurring-details/{purchaseId}/{packageName}                      | N/A                                                                                              | 월정액 상품 구매 상세조회 API 로 통합                                                                                                   |
| 변경           | 자동결제 해지             | /v2/purchase/manage-payment-status/{purchaseId}/{packageName}/cancel                | <p>/v6/apps/{packageName}/purchases/auto/products<br>/{productId}/{purchaseToken}/cancel</p>     | 성공처리 응답규격변경                                                                                                               |
| 변경           | 자동결제 해지취소           | /v2/purchase/manage-payment-status/{purchaseId}/{packageName}/reactivate            | <p>/v6/apps/{packageName}/purchases/auto/products<br>/{productId}/{purchaseToken}/reactivate</p> | 성공처리 응답규격변경                                                                                                               |
| 변경           | 구매취소 내역조회           | /v2/purchase/voided-purchases/{packageName}                                         | /v6/apps/{packageName}/voided-purchases                                                          | <p>API 사용성 향상을 위하여 구매취소내역 조회기준(startTime, endTime)을 구매일시에서 구매취소일시로 변경</p><p>응답규격에 purchaseToken, marketCode 추가</p>        |
| 변경           | 구매상품 소비             | /v2/purchase/consume/{purchaseId}/{packageName}                                     | <p>/v6/apps/{packageName}/purchases/all/products/<br>{productId}/{purchaseToken}/consume</p>     | 성공처리 응답규격변경                                                                                                               |
| 추가           | 구매 확인               | N/A                                                                                 | <p>/v6/apps/{packageName}/purchases/all/products/<br>{productId}/{purchaseToken}/acknowledge</p> | 구매 확인 API 신규 추가                                                                                                           |

\


#### PNS 메시지 규격 변경  <a href="#id-09.-pns" id="id-09.-pns"></a>

* 원화 외의 통화를 지원하기 위하여 결제금액(price)의 데이터 타입이 Number에서 String으로 변경하였습니다.
* 원화 외의 통화를 지원하기 위하여 결제금액의 통화코드(priceCurrencyCode)를 추가하였습니다.
* 원화 외의 통화를 지원하기 위하여 결제수단별금액(amount)의 데이터 타입이 Number에서 String으로 변경하였습니다.
* 응답 규격에 purchaseToken, environment, marketCode 필드가 추가되었습니다.&#x20;

상세한 규격은 PNS 메시지 상세 변경 내역에서 확인 할 수 있습니다. &#x20;

### **원스토어 인앱결제 라이브러리 API V5(SDK V17)**  <a href="#id-09.-apiv5-sdkv17" id="id-09.-apiv5-sdkv17"></a>

* API 버전은 개발사 애플리케이션의 'AndroidManifest.xml' 파일에 아래와 같이 명시되어야 합니다.
* 자세한 내용은 인앱결제 적용을 위한 사전준비 페이지의 'Android Manifest 파일 설정' 부분을 참고하시기 바랍니다.
*   API V5 (SDK V17)에서 개선된 사항은 다음과 같습니다.

    <table><thead><tr><th>항목</th><th width="160">세부 개선 항목</th><th width="565">설 명</th><th></th></tr></thead><tbody><tr><td><strong>API</strong></td><td>Consume 개념 도입</td><td>아이템 미지급 상황 발생 시 앱에서 복구할 수 있는 로직을 제공합니다.</td><td></td></tr><tr><td></td><td>Custom PID 지원</td><td>인앱상품 ID(PID)를 개발자가 직접 입력할 수 있어, 직관적인 PID로 생성하여 운영할 수 있습니다.</td><td></td></tr><tr><td></td><td>보편적인 인앱결제 인터페이스 제공</td><td>타 마켓향으로 개발한 앱을 코딩상의 큰 개발공수 없이 원스토어향으로 런칭할 수 있습니다.</td><td></td></tr><tr><td><strong>결제 테스트 환경</strong></td><td>개발/상용 테스트 환경 분리 제공</td><td><p>개발사의 검증 환경과 동일 수준의 원스토어 결제환경을 제공합니다.</p><p>개발사의 개발환경에서는 원스토어 Sandbox와 연동하여 개발/테스트 할 수 있고, 개발사의 상용환경에서는 원스토어 상용환경에서 실결제를 해볼 수 있습니다.</p><p>원스토어 개발자센터에서 테스트 ID로 등록한 ID로 원스토어 로그인 후, 실결제한 경우, 개발자가 자유롭게 구매취소 가능합니다.</p><p></p><table><thead><tr><th>구분</th><th></th><th>개발자</th><th>개발자</th><th data-hidden></th></tr></thead><tbody><tr><td></td><td></td><td>개발환경</td><td>상용환경</td><td></td></tr><tr><td>원스토어 결제환경</td><td>Sandbox</td><td>O</td><td>-</td><td></td></tr><tr><td></td><td>상용환경(실 결제)</td><td>-</td><td>O</td><td>O</td></tr></tbody></table></td><td></td></tr><tr><td><strong>결제창 UI</strong></td><td>전면/팝업 결제화면 선택 가능</td><td>전체 결제화면 또는 팝업 결제화면을 선택할 수 있습니다.<br>결제화면 설정은 원스토어 인앱결제 적용을 위한 사전준비 페이지의 'Android Manifest 파일 설정' 부분을 참고하시기 바랍니다.</td><td></td></tr><tr><td><strong>연동 방식</strong></td><td>IAP SDK, AIDL 선택 가능</td><td>기존의 SDK 방식 뿐만 아니라, AIDL를 이용하여 직접 OSS와 연동할 수 있습니다.<br>단, 원스토어 서비스(ONE store service, OSS) 앱이 설치되어 있지 않은 단말에서는 인앱결제가 정상적으로 동작하지 않습니다.<br>원스토어 서비스 앱 설치에 대한 내용은 원스토어 인앱결제 적용을 위한 사전준비 페이지의 '원스토어 앱 설치' 부분을 참고하시기 바랍니다.</td><td></td></tr><tr><td><strong>개발자센터</strong></td><td>타 마켓 앱 상품 정보 크롤링 기능 개선</td><td>기 등록된 타 마켓 앱 상품을 원스토어에 간편하게 등록할 수 있습니다.</td><td></td></tr><tr><td></td><td>타 마켓 인앱상품 정보 import 기능 제공</td><td></td><td></td></tr><tr><td><strong>인증센터</strong></td><td>OAuth 발급/인증/관리</td><td>개발자의 서버와 원스토어 인앱결제 서버 연동 시 데이터 보안을 강화하였습니다.</td><td></td></tr></tbody></table>

\


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/sample.md
# 10. Sample App Download

### **원스토어 인앱결제 Sample App 다운로드** <a href="#id-10.sampleappdownload-sampleapp" id="id-10.sampleappdownload-sampleapp"></a>

원스토어 인앱결제 API V7(SDK V21)를 사용하기 위해 필요한 샘플 앱을 [깃허브(github](https://github.com/ONE-store/onestore_iap_release))에서 다운로드 받을 수 있습니다.&#x20;

\


| branch 명 | Description                                                                                                                                                                                                                                                                                                                  |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| master   | <p><a href="https://github.com/ONE-store/onestore_iap_release/tree/master/onestore_iap_sample/sample_luckyone">sample_luckyone </a>: 인앱을 적용한 샘플앱<br><a href="https://github.com/ONE-store/onestore_iap_release/tree/master/onestore_iap_sample/sample_subscription">sample_subscription</a> : 구독상품 적용과 관리화면 연동을 보여주는 샘플앱</p> |

\


\


\


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/upgrade.md
# 11. V21로 원스토어 인앱결제 업그레이드 하기

### v19에서 v21로 업그레이드 하기 <a href="#id-11.v21-v19-v21" id="id-11.v21-v19-v21"></a>

#### 결제 라이브러리 변경하기 <a href="#id-11.v21" id="id-11.v21"></a>

원스토어 InApp Purchase(IAP) SDK는 v21버전부터 maven을 통한 배포를 제공하고 있습니다.

maven을 통해 IAP library를 사용하고 싶은 경우 다음의 내용에 따라 변경을 해야합니다.

* v19 AAR 파일 삭제
  * Project의 libs 폴더의 iap\_sdk-v19.XX.XX.aar 파일을 삭제합니다.&#x20;
* &#x20;maven 종속성 추가
  *   프로젝트 최상위 _build.gradle_ 파일에 원스토어 maven 주소를 등록합니다.

      ```gradle

      allprojects {
      	repositories {
      		...
      		maven { url 'https://repo.onestore.net/repository/onestore-sdk-public' }
      	}
      }
      ```
  *   다음은 앱의 _build.gradle_ 파일에 원스토어 결제 라이브러리 종속 항목을 추가합니다.

      ```gradle

      dependencies {
          def onestore_iap_version = "21.00.00"
          implementation "com.onestorecorp.sdk:sdk-iap:$onestore_iap_version"
      }

      ```

#### Json file 변경하기 <a href="#id-11.v21-jsonfile" id="id-11.v21-jsonfile"></a>

원스토어는 IAP SDK v19부터 SDK에서 사용하기 위한 필수 param들을 global-appstores.json을 통해 제공하였습니다.

v21부터는 좀더 편리하게 해당 값을 적용하기 위해 maven으로 해당 값을 배포하고 있습니다.

해당 내용을 변경하기 위해서는 다음을 따라 주세요.

* global-appstores.json 삭제
  * Project의 assets 폴더의 global-appstores.json파일을 삭제합니다.&#x20;
* &#x20;maven 종속성 추가
  *   프로젝트 최상위 _build.gradle_ 파일에 원스토어 maven 주소를 등록합니다.

      ```gradle

      allprojects {
          repositories {
              ...
              maven { url 'https://repo.onestore.net/repository/onestore-sdk-public' }
          }
      }

      ```
  *   다음은 앱의 _build.gradle_ 파일에 원스토어 결제 라이브러리 종속 항목을 추가합니다.

      ```gradle

      dependencies {
          def onestore_configuration_version = "1.0.0"
          implementation "com.onestorecorp.sdk:sdk-configuration-kr:$onestore_configuration_version"
      }

      ```

#### Deprecated 항목 수정하기 <a href="#id-11.v21-deprecated" id="id-11.v21-deprecated"></a>

v21 SDK에서 다음의 항목들이 Deprecated 되었습니다.

사용하는데는 문제는 없으나, 추후 원활한 사용을 위해 변경이 필요합니다.

*   Interface

    | name              | description                                                   |
    | ----------------- | ------------------------------------------------------------- |
    | PurchasesListener | v21부턴 ~~PurchasesListener~~ 대신 QueryPurchasesListener를 사용하세요. |
*   Method

    <table><thead><tr><th>class</th><th>method name</th><th>description</th></tr></thead><tbody><tr><td><pre><code>PurchaseClient
    </code></pre><p><br><br></p></td><td>manageRecurringProductAsync</td><td><p>v21부터 Auto(월정액) 상품 대신 Subscription(구독) 상품을 지원하며 정기 결제 메뉴를 통해 해당 기능들을 지원합니다.</p><p>정기 결제 메뉴의 경우 월정액 상품을 지원하지 않기 때문에 해당 API를 이용하여 이용해지 예약/취소를 기존처럼 제공해야 합니다.</p><p>구독 상품의 경우 정기 결제 메뉴로 사용자에게 안내하도록 구현이 필요합니다.</p></td></tr><tr><td>launchLoginFlowAsync</td><td><p>인증 관련 flow는 별도의 class로 분리하였습니다.</p><p>자세한 사항은 <a href="../sdk#id-04.-sdk-5">원스토어 로그인하기</a>를 참고하시기 바랍니다.</p></td><td></td></tr></tbody></table>

### v17에서 v21로 업그레이드 하기 <a href="#id-11.v21-v17-v21" id="id-11.v21-v17-v21"></a>

#### 결제 라이브러리 변경하기 <a href="#id-11.v21-.1" id="id-11.v21-.1"></a>

v17에서 v21로 변경 시 라이브러리가 jar 에서 aar로 변경됩니다.

v17 라이브러리를 사용하는 개발사의 경우 v19에서 v21로 업그레이드를 가이드를 통해 maven으로 변경 후, jar 형태에서 aar배포로 바뀐 부분에 대한 추가적인 작업(meta-data 삭제)이 필요합니다.

* v17 jar 라이브러리 삭제
  * Project의 libs폴더의 jar파일을 삭제합니다.&#x20;
*   AndroidManifest의 다음의 meta-data 삭제

    hljs.highlightAll();

    ```xml
              
    <meta-data android:name="iap:api_version" android:value="5"/>

    ```
* maven 종속성 추가 : [v19에서 v21로 업그레이드 하기](upgrade)를 참고하세요. &#x20;

#### Json file 종속성 추가 <a href="#id-11.v21-jsonfile" id="id-11.v21-jsonfile"></a>

v19 이상 부터는 SDK의 동작을 위해 json 파일이 필요합니다. [v19에서 v21로 업그레이드 >  json file 변경하기 > maven종속성 추가](upgrade) 를 참고하여 Project에 종속성을 추가 해주세요.

#### 변경 사항 수정하기 <a href="#id-11.v21" id="id-11.v21"></a>

[원스토어 인앱결제 SDK 업그레이드 안내](../old-version/v19/undefined-6)를 참고하여 변경 사항을 수정해주세요.

\


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/unity.md
# 12. Unity에서 원스토어 인앱결제 (SDK V21) 사용하기

## 개요

원스토어 결제 플러그인은 Unity 환경에서 에셋을 확장하여 게임에서 원스토어 결제 라이브러리의 최신 기능을 제공합니다. 이 가이드에서는 프로젝트를 설정하여 플러그인을 사용하는 방법과 원스토어 결제 라이브러리 기능을 구현하는 방법을 설명합니다.

### 개발 버전

<table data-view="cards"><thead><tr><th></th><th></th></tr></thead><tbody><tr><td><strong>Unity</strong></td><td>2022.3.11f1</td></tr><tr><td><strong>Java SDK (Java 11)</strong></td><td><p>Purchase: v21.02.01</p><p>App License Checker: v2.2.1</p></td></tr></tbody></table>

## 원스토어 결제 플러그인 설정

### 플러그인 다운로드 및 가져오기

1. [GitHub의 Release 페이지](https://github.com/ONE-store/unity_plugins/releases)에서 Unity 용 원스토어 인앱결제 플러그인 최신 버전을 다운로드합니다.
2.  Unity 메뉴 바에서 **Assets > Import Package > Custom Package**를 클릭합니다.\


    <figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2FaDPUiCuuNT9GzJMLxE4f%2Fimage.png?alt=media&#x26;token=ec33961f-2730-433f-b3e3-02c785b0047a" alt=""><figcaption></figcaption></figure>
3. 다운로드한 위치를 찾아 .unitypackage 파일을 선택합니다.
4. **Import Unity Package** 대화상자에서 모든 에셋을 선택한 상태로 두고 **Import**를 클릭합니다.

패키지를 가져오면 폴더가 추가됩니다. 이 폴더에 원스토어 결제 라이브러리가 포함되어 있습니다.

* Assets/OneStoreCorpPlugins
  * /Common
  * /Authentication
  * /Purchase
  * /AppLicenseChecker

> [EDM4U(External Dependency Manager for Unity)](https://github.com/googlesamples/unity-jar-resolver)가 필수로 같이 배포됩니다.\
> 만약, 이미 사용하고 있다면 `Import Package` 단계에서 `ExternalDependencyManager`를 체크해지하고 적용하면 됩니다.

### 외부 종속성 포함

프로젝트에 레파지토리 및 종속성이 포함되도록 하려면 다음 단계를 따릅니다.

**`Project Settings > Player > Publishing Settings > Build`**

아래의 두 가지를 체크합니다.

* Custom Main Manifest
*   Custom Main Gradle Template\


    <figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2FwxTyCfDtU8RCaZVxrlUW%2Fimage.png?alt=media&#x26;token=155e429b-5b42-4e44-bf5f-b24fa4e5a789" alt=""><figcaption></figcaption></figure>

**`Assets > External Dependency Manager > Android Resolver > Force Resolve`** 를 선택합니다.

`mainTemplete.gradle` 파일에 아래와 같이 적용되는 것을 확인할 수 있습니다.\


<figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2F9Nl9wpIgGFtlWanaWbuM%2Fimage.png?alt=media&#x26;token=4f08e2d8-06bf-4dd8-b933-c8bb9ef15320" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
* **In-app Purchase 종속성**은 다음에 나열되어 있습니다.\
  Assets/OneStoreCorpPlugins/Purchase/Editor/PurchaseDependencies.xml
* **App License Checker 종속성**은 다음에 나열되어 있습니다.\
  Assets/OneStoreCorpPlugins/AppLicenseChecker/Editor/AppLicenseCheckerDependencies.xml
{% endhint %}

### \<queries> 설정

`AndroidManifest.xml` 파일에 `<queries>`를 설정 해야합니다.\
자세한 내용은 [공지사항](https://dev.onestore.co.kr/devpoc/support/news/noticeView.omp?noticeId=32968)을 참조하세요.

{% hint style="danger" %}
**\<queries>** 태그를 설정하지 않으면 SDK에서 원스토어 서비스를 찾을 수 없습니다.
{% endhint %}

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

### 스토어 지정을 위한 개발자 옵션 설정

&#x20;IAP SDK 21.02.00 부터 아래와 같이 `onestore:dev_option`의 `android:value` 값을 설정하면, SDK와 연동되는 스토어 앱을 지정 할 수 있습니다.

```xml
<manifest>
    <application>
        <meta-data android:name="onestore:dev_option" android:value="onestore_01" />
    </application>
</manifest>
```

<table><thead><tr><th width="226">값 (android:value)</th><th>적용 대상 국가</th></tr></thead><tbody><tr><td><code>onestore_00</code></td><td>대한민국 (South Korea) <em>(기본값)</em></td></tr><tr><td><code>onestore_01</code></td><td>싱가포르, 타이완 (Singapore, Taiwan)</td></tr><tr><td><code>onestore_02</code></td><td>미국 – Digital Turbine (United States)</td></tr></tbody></table>

{% hint style="warning" %}
21.01.00 버전에서는 android:value 값이 global만 설정 가능하며, 싱가포르/타이완 스토어 앱만 지정이 가능했습니다.&#x20;
{% endhint %}

{% hint style="danger" %}
**주의**: 배포 버전의 바이너리에서는 이 옵션을 반드시 제거해주세요.&#x20;
{% endhint %}

## 게임에서 원스토어 인앱 결제 라이브러리 적용하기

### 로그레벨 설정

개발 단계에서 로그 레벨을 설정하여 SDK의 데이터의 흐름을 좀 더 자세히 노출할 수 있습니다.\
[`android.util.Log`](https://developer.android.com/reference/android/util/Log#summary)에 정의된 값을 기반으로 동작합니다.

```csharp
using OneStore.Common;
/// <summary>
/// Warning! This code must be deleted during release build as it may be vulnerable to security.
/// Use <seealso cref="OneStoreLogger.SetLogLevel(int)"/> only for development.
/// </summary>
OneStoreLogger.SetLogLevel(2);
```

| 상수             | 값 |
| -------------- | - |
| VERBOSE        | 2 |
| DEBUG          | 3 |
| INFO (default) | 4 |
| WARN           | 5 |
| ERROR          | 6 |

{% hint style="danger" %}
배포 빌드 버전에서는 보안에 취약할 수 있으니 이 옵션을 **삭제**해야 합니다.
{% endhint %}

### 원스토어 인앱 결제 초기화

<figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2FUAnVFk26E1bMVQTJDMtk%2Fimage.png?alt=media&#x26;token=155562a8-146c-4e99-89ee-1ec5966676e8" alt=""><figcaption></figcaption></figure>

인앱결제를 요청하기 위해 원스토어 개발자 센터에서 제공하는 라이선스 키를 사용하여 `PurchaseClientImpl` 객체를 초기화를 합니다.  `Initialize()` 함수를 호출하여 기본적인 세팅을 마무리합니다. 이는 원스토어 서비스에 연결하기 위한 선행 작업입니다.

```csharp
using OneStore.Purchasing;

class YourCallback: IPurchaseCallback
{
    // IPurchaseCallback implementations.
}

var purchaseClient = new PurchaseClientImpl("your license key");
purchaseClient.Initialize(new YourCallback());
```

### 상품 정보 조회하기

`PurchaseClientImpl` 객체를 초기화 완료 후 `QueryProductDetails()`를 통해 상품 정보를 요청하면 `IPurchaseCallback.OnProductDetailsSucceeded()`로 응답을 받습니다.

`ProductType`은 아래와 같습니다.

<table><thead><tr><th width="246">Product</th><th>Enum</th></tr></thead><tbody><tr><td>관리형 상품</td><td><code>ProductType.INAPP</code></td></tr><tr><td>정기 결제 상품 (구독 상품)</td><td><code>ProductType.SUBS</code></td></tr><tr><td><del>월 정액 상품</del></td><td><del><code>ProductType.AUTO</code></del><br>(이 상품은 향후 지원되지 않을 예정입니다.) </td></tr></tbody></table>

위의 모든 유형의 데이터를 한 번에 조회하고 싶을 경우는 `ProductType.ALL` 설정하면 됩니다.

{% hint style="warning" %}
`ProductType.ALL`은 [상품 정보 조회하기](#undefined-9)에서만 사용할 수 있으며, [구매 요청하기](#undefined-10), [구매 내역 조회하기](#id-12.unity-sdkv21-4)에서는 사용할 수 없습니다.
{% endhint %}

```csharp
using OneStore.Purchasing;
  
List items = ...
purchaseClient.QueryProductDetails(items.AsReadOnly(), ProductType.INAPP);

// IPurchaseCallback implementations
public void OnProductDetailsSucceeded(List productDetails)
{
    ...
}

public void OnProductDetailsFailed(IapResult iapResult)
{
    ...
}
```

{% hint style="warning" %}
인앱 상품 ID 목록은 개발사의 보안 백엔드 서버에서 관리해야 합니다.
{% endhint %}

### 구매 요청하기

앱에서 구매 요청을 하기 위해서는 기본 스레드에서 `Purchase()` 함수를 호출합니다.

`QueryProductDetails()` API를 호출해서 얻은 `ProductDetail` 객체의 값을 토대로 `PurchaseFlowParams` 객체를 생성합니다.\
`PurchaseFlowParams` 객체를 생성하려면 `PurchaseFlowParams.Builder` 클래스를 사용합니다.

`SetDeveloperPayload()`는 개발사에서 임의로 입력한 값으로 최대 `200byte`입니다. 이 값은 결제 후에 데이터의 정합성과 부가 데이터를 확인하기 위해 사용할 수 있습니다.\
`SetProductName()`은 상품 이름을 결제 시 변경하여 노출하고 싶을 때 사용됩니다.\
`SetQuantity()`는 관리형 인앱 상품에만 적용되며 한 상품을 여러 개 구매할 때 사용됩니다.

{% hint style="success" %}
원스토어는 사용자에게 할인 쿠폰, 캐시백 등의 다양한 혜택 프로모션을 진행하고 있습니다.\
개발사는 구매 요청 시에 `gameUserId`, `promotionApplicable` 파라미터를 이용하여 앱을 사용하는 유저의 프로모션 참여를 제한하거나 허용할 수 있습니다.\
개발사는 앱의 고유한 유저 식별 번호 및 프로모션 참여 여부를 선택하여 전달하고, 원스토어는 이 값을 토대로 사용자의 프로모션 혜택을 적용하게 됩니다.
{% endhint %}

{% hint style="warning" %}
`gameUserId`, `promotionApplicable` 파라미터는 옵션 값으로 원스토어 사업 부서 담당자와 프로모션에 대해 사전협의가 된 상태에서만 사용하여야 하며, 일반적인 경우에는 값을 보내지 않습니다.\
또한, 사전협의가 되어 값을 보낼 경우에도 개인 정보보호를 위해 `gameUserId`는 `hash`된 고유한 값으로 전송하여야 합니다.
{% endhint %}

```csharp
  using OneStore.Purchasing;
  
  ProductDetail productDetail = ...
  ProductType productType = ProductType.Get(productDetail.type);

  var purchaseFlowParams = new PurchaseFlowParams.Builder()
          .SetProductId(productId)                // mandatory
          .SetProductType(productType)            // mandatory
          .SetDeveloperPayload(developerPayload)  // optional
          .SetQuantity(quantity)                  // optional
          // .SetProductName(null)                // optional: Change the name of the product to appear on the purchase screen.
          
          // It should be used only in advance consultation with the person in charge of the One Store business, and is not normally used.
          // .SetGameUserId(null)                 // optional: User ID to use for promotion.
          // .SetPromotionApplicable(false)       // optional: Whether to participate in the promotion.
          .Build(); 

  purchaseClient.Purchase(purchaseFlowParams);
```

`Purchase()` 호출에 성공하면 아래와 같은 화면을 표시합니다. **\[그림 1]**&#xC740; 관리형 상품 구매 화면을 나타냅니다.\


<figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2FQlQqSb2MeYLL9LZvqvAT%2Fimage.png?alt=media&#x26;token=dfbde249-6cb3-461e-abc8-a8d830f41d46" alt=""><figcaption><p>[그림 1] </p></figcaption></figure>

구매에 성공하면 `IPurchaseCallback.OnPurchaseSucceeded()` 함수에 결과를 전송합니다.\
실패 시에는 `IPurchaseCallback.OnPurchaseFailed()` 함수가 호출됩니다.

```csharp
  using OneStore.Purchasing;
  
  // IPurchaseCallback implementations
  public void OnPurchaseSucceeded(List purchases)
  {
      handlePurchase(purchases);
  }

  public void OnPurchaseFailed(IapResult iapResult)
  {
      ...
  }
```

\
구매가 성공하면 구매 데이터에는 사용자 및 상품 ID를 나타내는 고유 식별자인 구매 토큰도 생성됩니다. 구매 토큰을 앱 내에 저장할 수 있지만 구매를 인증하고 사기로부터 보호할 수 있는 백엔드 서버로 토큰을 전달하는 것이 좋습니다.

<figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2F7QWzCSwkc5tCI8ZTdTqM%2Fimage.png?alt=media&#x26;token=a5d9347e-c6b9-4b60-bc37-1ddbeae24270" alt=""><figcaption></figcaption></figure>

관리형 상품과 정기 결제 상품의 구매 토큰은 결제가 일어날 때마다 구매 토큰이 발행됩니다. (월 정액 상품의 경우 자동 결제가 갱신되는 동안 구매 토큰은 동일하게 유지됩니다.)

또한 사용자는 영수증 번호가 포함된 거래 영수증을 이메일로 받습니다. 관리형 상품은 구매할 때마다 이메일을 받으며, 월 정액 상품과 정기 결제 상품은 처음 구매 시 그리고 이후 갱신될 때마다 이메일을 받습니다.

### 정기 결제 <a href="#id-12.unity-sdkv21" id="id-12.unity-sdkv21"></a>

정기 결제는 취소될 때까지 자동으로 갱신됩니다. 정기 결제는 다음 상태를 가질 수 있습니다.

* **활성**: 사용자가 콘텐츠 사용에 문제가 없는 양호한 상태이며 정기 결제에 접근할 수 있습니다.
* **일시 중지 예약**: 사용자가 정기 결제를 이용 중 일시 중지를 하고 싶을 때 선택할 수 있습니다.
  * 주간 정기 결제: 1\~3주 단위로 일시 중지할 수 있습니다.
  * 월간 정기 결제: 1\~3개월 단위로 일시 중지할 수 있습니다.
  * 연간 정기 결제: 일시 중지를 지원하지 않습니다.
* **해지 예약**: 사용자가 정기 결제를 이용 중이지만 취소하고 싶을 때 선택할 수 있습니다. 다음 결제일에 결제가 되지 않습니다.
* **유예, 보류**: 사용자에게 결제 문제가 발생하면 다음 결제일에 결제가 되지 않습니다. 취소 예약을 할 수 없으며 즉시 "구독 해지"를 할 수 있습니다.

#### 사용자가 정기 결제를 업그레이드, 다운그레이드 또는 변경할 수 있도록 허용 <a href="#id-12.unity-sdkv21" id="id-12.unity-sdkv21"></a>

사용자가 정기 결제를 업그레이드하거나 다운그레이드 하려면 구매 시 _비례 배분 모&#xB4DC;_&#xB97C; 설정하거나 변경사항이 정기 결제 사용자에게 영향을 주는 방식을 설정할 수 있습니다.\
다음 표에는 사용 가능한 `비례 배분 모드`_(_`OneStoreProrationMode`_)_&#xAC00; 나와 있습니다.

| **비례 배분 모드**                            | **설명**                                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------------------------- |
| IMMEDIATE\_WITH\_TIME\_PRORATION        | 정기 결제의 교체가 즉시 이루어지며, 남은 시간은 가격 차이를 기반으로 조정되어 입금되거나 청구됩니다. (이것은 기본 동작입니다.)                   |
| IMMEDIATE\_AND\_CHARGE\_PRORATED\_PRICE | 정기 결제의 교체가 즉시 이루어지며, 청구 주기는 동일하게 유지됩니다. 나머지 기간에 대한 가격이 청구됩니다. (이 옵션은 업그레이드 에서만 사용할 수 있습니다.) |
| IMMEDIATE\_WITHOUT\_PRORATION           | 정기 결제의 교체가 즉시 이루어지며, 다음 결제일에 새로운 가격이 청구됩니다. 청구 주기는 동일하게 적용됩니다.                              |
| DEFERRED                                | 기존 요금제가 만료되면 교체가 적용되며 새 요금이 동시에 청구됩니다.                                                      |

#### 업그레이드 또는 다운그레이드 <a href="#id-12.unity-sdkv21" id="id-12.unity-sdkv21"></a>

정기 결제는 [구매 요청하기](#undefined-10)와 동일한 API를 사용하여 사용자에게 업그레이드 또는 다운그레이드를 제공할 수 있습니다. 다만, 정기 결제의 업그레이드 다운그레이드를 적용하기 위해선 기존 정기 결제 구매 토큰과 비례 배분 모드 값이 필수로 필요합니다.\
다음 예와 같이 현재 정기 결제, 향후(업그레이드 또는 다운그레이드) 정기 결제 및 비례 배분 모드에 관한 정보를 제공해야 합니다.

```csharp
using OneStore.Purchasing;
  
public void UpdateSubscription(ProductDetail productDetail, PurchaseData oldPurchase, OneStoreProrationMode mode, string developerPayload)
{
    ProductType productType = ProductType.Get(productDetail.type);
  
    var purchaseFlowParams = new PurchaseFlowParams.Builder()
             .SetProductId(productId)                        // mandatory
             .SetProductType(productType)                    // mandatory
             .SetOldPurchaseToken(oldPurchase.PurchaseToken) // mandatory
             .SetProrationMode(mode)                         // mandatory
  
             .SetDeveloperPayload(developerPayload)          // optional
             // .SetProductName(null)                        // optional: Change the name of the product to appear on the purchase screen.
             .Build(); 

    purchaseClient.UpdateSubscription(purchaseFlowParams);
}
```

업그레이드 또는 다운그레이드의 경우도 [구매 요청하기](#undefined-10) 로직을 수행하기 때문에 응답은 똑같이 수신합니다.\
또한 [구매 내역 조회하기](#id-12.unity-sdkv21-4)에서도 요청 시 응답을 받을 수 있습니다. 비례 배분 모드로 구매했을 때도 일반 구매와 동일하게 `PurchaseClientImpl.AcknowledgePurchase()`를 사용하여 [구매 후 처리](#id-12.unity-sdkv21-3)를 해야 합니다.

### 구매 후 처리 <a href="#id-12.unity-sdkv21" id="id-12.unity-sdkv21"></a>

사용자가 구매를 완료하면 앱에서 구매를 처리해야 합니다. 대부분의 경우 앱은 `OnPurchaseSucceeded()`를 통해 구매 알림을 받습니다. 또는  [구매 내역 조회하기](#id-12.unity-sdkv21-4)에서 설명된 것처럼 앱이 `PurchaseClientImpl.QueryPurchases()` 함수를 호출하여 처리하는 경우가 있습니다.

#### 구매 소비하기 (consume) <a href="#id-12.unity-sdkv21-consume" id="id-12.unity-sdkv21-consume"></a>

관리형 상품은 소비를 하기 전까지는 재 구매할 수 없습니다.

상품을 소비하기 위해서는 `ConsumePurchase()`를 호출합니다. 또한 소비 작업 결과는 `IPurchaseCallback..OnConsumeSucceeded()` 호출 됩니다.

{% hint style="info" %}
관리형 상품을 소비하지 않으면 영구성 형태의 상품 타입처럼 활용할 수 있으며, 구매 후 즉시 소비하면 소비성 형태의 상품으로도 활용됩니다.\
또한 특정 기간 이후에 소비하면 기간제 형태의 상품으로 활용할 수 있습니다.
{% endhint %}

```csharp
using OneStore.Purchasing;

public void handlePurchase(PurchaseData purchaseData)
{
    purchaseClient.ConsumePurchase(purchaseData);
}

// IPurchaseCallback implementations
public void OnConsumeSucceeded(PurchaseData purchase)
{
    ...
}

public void OnConsumeFailed(IapResult iapResult)
{
    ...
}
```

{% hint style="info" %}
소비 요청이 때로 실패할 수 있으므로 보안 백엔드 서버를 확인하여 각 구매 토큰이 사용되지 않았는지 확인해야 합니다. 그래야 앱이 동일한 구매에 대해 여러 번 자격을 부여하지 않습니다. 또는 자격을 부여하기 전에 성공적인 소비 응답을 받을 때까지 기다릴 수 있습니다.
{% endhint %}

{% hint style="danger" %}
3일 이내에 구매를 확인(`acknowledge`) 또는 소비(`consume`)를 하지 않으면 사용자에게 상품이 지급되지 않았다고 판단되어 자동으로 환불됩니다.
{% endhint %}

#### 구매 확인하기 (acknowledge) <a href="#id-12.unity-sdkv21-acknowledge" id="id-12.unity-sdkv21-acknowledge"></a>

비 소비성 상품을 인증하려면 `PurchaseClientImpl.AcknowledgePurchase()` 함수를 사용합니다. 관리형 상품, 월 정액 상품, 정기 결제 상품 모두 사용할 수 있습니다.

`PurchaseData.Acknowledged()` 함수를 사용하여 인증 되었는지를 판단할 수 있습니다. 또한 인증 작업 결과가 성공하면

`IPurchaseCallback.OnAcknowledgeSucceeded()` 함수가 호출됩니다.

```csharp
using OneStore.Purchasing;

public void handlePurchase(PurchaseData purchaseData)
{
    if (!purchaseData.Acknowledged())
    {
         purchaseClient.AcknowledgePurchase(purchaseData);
    }
}

// IPurchaseCallback implementations
public void OnAcknowledgeSucceeded(PurchaseData purchase)
{
    ...
}

public void OnAcknowledgeFailed(IapResult iapResult)
{
    ...
}
```

### 구매 내역 조회하기 <a href="#id-12.unity-sdkv21" id="id-12.unity-sdkv21"></a>

구매를 처리하는 것만으로는 앱이 모든 구매를 처리하는 것을 보장하기에 충분하지 않습니다. 앱에서 사용자가 구매한 모든 항목을 인식하지 못할 수 있습니다. 앱에서 구매 추적을 놓치거나 구매를 인식하지 못할 수 있는 몇 가지 시나리오는 다음과 같습니다.

* **구매 중 네트워크 문제**: 사용자가 구매를 성공적으로 완료하고 원스토어에서 확인을 받았지만 기기가 구매 알림을 받기 전에 네트워크 연결이 끊어졌을 경우
* **여러 기기**: 사용자는 한 기기에서 항목을 구입한 후 기기를 전환할 때 이 항목이 표시되기를 기대합니다.

이러한 상황을 처리하려면 앱의 `Start()` 또는 `OnApplicationPause()`에서 `PurchaseClientImpl.QueryPurchases()`를 호출하여 `구매 후 처리`에 설명된 대로 모든 구매가 성공적으로 처리되었는지 확인해야 합니다.

```csharp
using OneStore.Purchasing;
  
// IPurchaseCallback implementations
public void OnPurchaseSucceeded(List purchases)
{
    handlePurchase(purchases);
}

public void OnPurchaseFailed(IapResult iapResult)
{
    ...
}
```

### 월 정액 상품 상태 변경하기 (Deprecated) <a href="#id-12.unity-sdkv21-deprecated" id="id-12.unity-sdkv21-deprecated"></a>

월 정액 상품은 최초 구매 후 30일 갱신이 이루어지는 상품입니다. 월 정액 상품의 상태는 `PurchaseData.RecurringState()`를 통해 확인할 수 있습니다.

월 정액 상품의 상태를 변경하려면 `PurchaseClientImpl.ManageRecurringProduct()`를 사용합니다.\
구매 데이터와 변경하려는 `RecurringAction` 값을 입력합니다.

```csharp
using OneStore.Purchasing;
  
RecurringAction recurringAction = purchaseData.RecurringState == 0 ? RecurringAction.CANCEL : RecurringAction.REACTIVATE;
purchaseClient.ManageRecurringProduct(purchaseData, recurringAction); 

// IPurchaseCallback implementations
public void OnManageRecurringProduct(IapResult iapResult, PurchaseData purchase, RecurringAction action)
{
     if (iapResult.IsSuccessful())
    {
        ...
    }
    else
    {   
        ...
    }
}
```

### 정기 결제 관리 화면을 열기 <a href="#id-12.unity-sdkv21" id="id-12.unity-sdkv21"></a>

구독한 상품의 상태를 관리하는 화면을 띄울 수 있습니다.\
매개변수로 `PurchaseData`를 넣으면 구매 데이터를 확인하여 해당 정기 결제 상품의 관리 화면을 실행합니다. 그러나 `null` 넣을 경우 사용자의 정기 결제 리스트 화면을 실행합니다.\
다음은 정기 결제 관리 화면을 띄우는 방법을 나타내는 예제입니다.

```csharp
using OneStore.Purchasing;
  
PurchaseData purchaseData = ...;
purchaseClient.LaunchManageSubscription(purchaseData); 
```

### 마켓 구분 코드 얻기 <a href="#id-12.unity-sdkv21" id="id-12.unity-sdkv21"></a>

SDK v19 이상부터 S2S API를 사용하기 위해서는 마켓 구분 코드가 필요합니다.

`PurchaseClientImpl` 객체를 초기화시 SDK는 결제 모듈과 연결을 시도합니다. 이 때 연결이 성공적으로 완료되면 _자동으로_ `StoreCode`를 가져옵니다.

`PurchaseClientImpl.StoreCode` 변수에 할당되어 있습니다.

```csharp
using OneStore.Purchasing;
  
// Possible after calling the PurchaseClientImpl.RetrieveProducts()
var storeCode = purchaseClient.StoreCode;
```

### 설치 마켓 확인하기&#x20;

`StoreEnvironment.GetStoreType()` API는 SDK가 탑재된 애플리케이션이 원스토어를 통해 설치되었는지를 판단하는 기능을 제공합니다.

#### Store Type 정의

해당 API는 `StoreType`을 반환하며, 아래 네 가지 값 중 하나를 가집니다.

<table><thead><tr><th width="240">StoreType</th><th width="71">value</th><th>description</th></tr></thead><tbody><tr><td><code>StoreType.UNKNOWN</code></td><td>0</td><td>앱 설치 스토어 정보를 알 수 없음 <em>(APK 직접 설치, 출처 불명 등)</em></td></tr><tr><td><code>StoreType.ONESTORE</code></td><td>1</td><td>ONE Store에서 설치됨 <em>(또는 개발자 옵션이 활성화된 경우)</em></td></tr><tr><td><code>StoreType.VENDING</code></td><td>2</td><td>Google Play Store에서 설치됨</td></tr><tr><td><code>StoreType.ETC</code></td><td>3</td><td>기타 스토어에서 설치됨</td></tr></tbody></table>

#### API 사용 방법

해당 API는 `StoreEnvironment.GetStoreType()`을 호출하여 사용할 수 있습니다.

```csharp
using OneStore.Common;

public void DetectStoreEnvironment()
{
    var storeType = StoreEnvironment.GetStoreType();
    switch (storeType) 
    {
        case StoreType.ONESTORE:
            UnityEngine.Debug.Log("ONE Store");
            break;
        case StoreType.VENDING:
            UnityEngine.Debug.Log("Google Play Store");
            break;
        case StoreType.ETC: 
            UnityEngine.Debug.Log("Other stores");
            break;
        // StoreType.UNKNOWN
        default:
            UnityEngine.Debug.Log("Unknown store");
            break;
    }
}
```

#### 스토어 판단 기준

이 API는 세 가지 방법을 통해 설치된 스토어를 판별합니다.

1. &#x20;원스토어 마켓 서명을 통해 배포된 경우
   1. 원스토어의 마켓 서명을 통한 배포 여부를 확인하여, 원스토어에서 설치된 앱인지 확인합니다.
2. Installer Package Name을 기반으로 판별
   1. 원스토어의 마켓 서명을 통해 배포되지 않은 경우, PackageManager.getInstallerPackageName() API를 이용하여 앱 설치 시 사용된 스토어 정보를 확인합니다.
3. 개발자 옵션(`onestore:dev_option`)이 활성화된 경우
   1. onestore:dev\_option 이 설정 되어있으면 무조건 StoreType.ONESTORE로 응답합니다.

### 원스토어 서비스 설치하기 <a href="#id-12.unity-sdkv21" id="id-12.unity-sdkv21"></a>

원스토어 서비스의 버전이 낮거나 없을 경우 인앱결제를 이용할 수 없습니다. _최초 API 호출 시_  우선 원스토어 서비스와 연결 시도를 먼저 합니다. 이때 `RESULT_NEED_UPDATE`가 발생하면 `LaunchUpdateOrInstallFlow()` 메서드를 호출해야 합니다.

```csharp
using OneStore.Purchasing;

purchaseClient.LaunchUpdateOrInstallFlow(); 
```

### 원스토어 로그인 요청하기  <a href="#id-12.unity-sdkv21" id="id-12.unity-sdkv21"></a>

원스토어 인앱 SDK는 사용자가 원스토어에 로그인이 되어 있어야 동작합니다. 내부적으로 우선 로그인 토큰으로 로그인을 시도합니다. 실패하거나 최초 로그인같이 사용자의 정보가 없을 경우 포그라운드 로그인 화면을 띄워 사용자로 하여금 로그인을 유도합니다.

```csharp
using OneStore.Auth;
  
new OneStoreAuthClientImpl().LaunchSignInFlow({signInResult} => {
    if (signInResult.IsSuccessful())
    {
        ...
    }
});
```


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/unitymig.md
# 13. Unity에서 IAP SDK v21로 업그레이드 하기

### **Unity 용 v21 라이브러리 파일 업데이트하기** <a href="#id-13.unity-iapsdkv21-unity-v21" id="id-13.unity-iapsdkv21-unity-v21"></a>

{% hint style="info" %}
단, 라이브러리 파일만 교체했을 경우 IAP SDK v21에서 새롭게 제공하는 복수 구&#xB9E4;_**,**_ 구독  등의 기능은 사용할 수 없습니다.
{% endhint %}

#### v19 라이브러리 파일 삭제하기 <a href="#id-13.unity-iapsdkv21-v19" id="id-13.unity-iapsdkv21-v19"></a>

| Assets > Plugins > Android | <ul><li>iap_sdk-v19.00.xx.aar</li><li>iap_adapter-v1.x.x.aar</li></ul> |
| -------------------------- | ---------------------------------------------------------------------- |
| Assets > StreamingAssets   | <ul><li>global-appstores.json</li></ul>                                |

#### AndroidManifest.xml에서 엘리먼트 제거하기 <a href="#id-13.unity-iapsdkv21-androidmanifest.xml" id="id-13.unity-iapsdkv21-androidmanifest.xml"></a>

Assets > Plugins > Android > AndroidManifest.xml

ProxyActivity 와 iap\_version 메타 데이터 제거합니다.

```xml

<manifest>
    <application>
         <activity android:name="com.gaa.sdk.iap.ProxyActivity"
                  android:configChanges="locale|fontScale|keyboard|keyboardHidden|layoutDirection|mcc|mnc|navigation|orientation|screenLayout|screenSize|smallestScreenSize|touchscreen|uiMode"
                  android:theme="@android:style/Theme.Translucent.NoTitleBar.Fullscreen"/>

         <meta-data android:name="iap:api_version" android:value="6"/>
        ...
    </application>
</manifest>

```

#### [Unity용 v21 라이브러리 파일](https://github.com/ONE-store/unity_plugins/tree/archive)을 추가하기  <a href="#id-13.unity-iapsdkv21-unity-v21" id="id-13.unity-iapsdkv21-unity-v21"></a>

아래의 폴더에 라이브러리 파일을 추가합니다.

| Assets > Plugins > Android | <ul><li>sdk-base-v1.0.0.aar</li><li>sdk-auth-v1.0.1.aar</li><li>sdk-iap-v21.00.00.aar</li><li>sdk-configuration-kr-v1.0.0.aar</li><li>iap-unity-adapter-v2.0.0.aar</li></ul> |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

\


\


### IAP SDK v21의 새로운 기능을 사용하기 위해 업그레이드하기 <a href="#id-13.unity-iapsdkv21-iapsdkv21" id="id-13.unity-iapsdkv21-iapsdkv21"></a>

#### v19 ".unitypackage"에서 추가되었던 파일 삭제하기 <a href="#id-13.unity-iapsdkv21-v19-.unitypackage" id="id-13.unity-iapsdkv21-v19-.unitypackage"></a>

| Assets > Plugins > Android  | <ul><li>iap_sdk-v19.00.xx.aar</li><li>iap_adapter-v1.x.x.aar</li></ul>                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Assets > Scripts > Purchase | <ul><li>GaaIapCallManager.cs</li><li>GaaIapCallbackManager.cs</li><li>GaaIapResultListener.cs</li><li>GaaIapPurchaseResponse.cs</li><li>AndroidNative.cs</li></ul> |
| Assets > StreamingAssets    | <ul><li>global-appstores.json</li></ul>                                                                                                                            |

#### AndroidManifest.xml에서 엘리먼트 제거하기 <a href="#id-13.unity-iapsdkv21-androidmanifest.xml-.1" id="id-13.unity-iapsdkv21-androidmanifest.xml-.1"></a>

Assets > Plugins > Android > AndroidManifest.xml

ProxyActivity 와 iap\_version 메타 데이터 제거합니다.

```xml

<manifest>
    <application>
         <activity android:name="com.gaa.sdk.iap.ProxyActivity"
                  android:configChanges="locale|fontScale|keyboard|keyboardHidden|layoutDirection|mcc|mnc|navigation|orientation|screenLayout|screenSize|smallestScreenSize|touchscreen|uiMode"
                  android:theme="@android:style/Theme.Translucent.NoTitleBar.Fullscreen"/>

         <meta-data android:name="iap:api_version" android:value="6"/>
        ...
    </application>
</manifest>

```

#### GameObject 제거하기 <a href="#id-13.unity-iapsdkv21-gameobject" id="id-13.unity-iapsdkv21-gameobject"></a>

GaaIapCallbackManager 게임 오브젝트 제거합니다.\
v21이 적용된 Unity Plugin에서는 이제 GameObject를 수동으로 추가할 필요가 없습니다.

<figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2FdGcTYMjcwzbmDh2CQfYm%2Fimage.png?alt=media&#x26;token=fff19c74-c179-4762-b9ac-c98fbada7c5d" alt=""><figcaption></figcaption></figure>

위의 모든 작업을 수행하였다면 IAP SDK v21을 적용하기에 앞서 사전 준비가 완료되었습니다.

이제 12. Unity에서 원스토어 인앱결제 사용하기 가이드를 참조하여 적용하시면 됩니다.



출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/flutter.md
# 14. Flutter에서 원스토어 인앱 결제 사용하기

## 개요 <a href="#undefined" id="undefined"></a>

Flutter 환경에서 구현된 어플리케이션에서 원스토어 결제 라이브러리의 최신 기능 제공합니다. 이 가이드는 원스토어 결제 라이브러리 기능을 Flutter 환경에서 적용하는 방법을 설명합니다.

## 개발 버전 <a href="#undefined-1" id="undefined-1"></a>



<table data-view="cards"><thead><tr><th></th><th></th><th></th></tr></thead><tbody><tr><td><strong>Flutter</strong></td><td>3.3.0</td><td></td></tr><tr><td><strong>Java SDK (Java 11)</strong></td><td><p>Purchase: v21.02.01</p><p>App License Checker: v2.2.1</p></td><td></td></tr><tr><td></td><td></td><td></td></tr></tbody></table>

## 플러그인 설정 <a href="#undefined-2" id="undefined-2"></a>

### pubspec.yaml 파일에 플러그인 추가하기 <a href="#pubspec.yaml" id="pubspec.yaml"></a>

`flutter pub get` 을 통하여 패키지를 다운로드합니다.

```
dependencids:
  flutter_onestore_inapp: ^0.3.0
```

### Android build.gradle에 종속성 추가하기 <a href="#android-build.gradle" id="android-build.gradle"></a>

Top-Level build.gradle 파일에 maven repository 주소 추가하기

```
allprojects {
    repositories {
        maven { url 'https://repo.onestore.net/repository/onestore-sdk-public' }
    }
}
```

### AndroidMainifest.xml 파일에 \<queries> 추가하기 <a href="#androidmainifest.xml-less-than-queries-greater-than" id="androidmainifest.xml-less-than-queries-greater-than"></a>

* 참고: [https://developer.android.com/training/package-visibility](https://developer.android.com/training/package-visibility)

`<manifest>` 태그의 직속 하위에 위치하며, 아래와 같은 요소를 추가해야 합니다.

```xml
<manifest>

    <queries>
        <intent>
            <action android:name="com.onestore.ipc.iap.IapService.ACTION" />
        </intent>
        <intent>
            <action android:name="android.intent.action.VIEW" />

            <data android:scheme="onestore" />
        </intent>
    </queries>

    <application>

    </application>
</manifest>
```

### 스토어 지정을 위한 개발자 옵션 설정 <a href="#undefined-3" id="undefined-3"></a>

#### v21.02.00 업데이트 – 글로벌 스토어 선택 기능 추가

IAP SDK 21.02.00 부터 아래와 같이 `onestore:dev_option`의 `android:value` 값을 설정하면, SDK와 연동되는 스토어 앱을 지정 할 수 있습니다.

`<application>` 태그 직속 하위에 위치하며, 아래와 같은 요소를 추가합니다.

```xml
<manifest>
    <application>
        <activity>
        </activity>
            <meta-data android:name="onestore:dev_option" android:value="onestore_01" />
    </application>
</manifest>
```

<table><thead><tr><th>값 (android:value)</th><th> 적용 대상 국가/지역</th><th data-hidden></th></tr></thead><tbody><tr><td>onestore_00</td><td>대한민국 (South Korea) <em>(기본값)</em></td><td></td></tr><tr><td>onestore_01</td><td>싱가포르, 타이완 (Singapore, Taiwan)</td><td></td></tr><tr><td>onestore_02</td><td>미국 – Digital Turbine (United States)</td><td></td></tr></tbody></table>

{% hint style="warning" %}
21.01.00 버전에서는 android:value 값이 global만 설정 가능하며, 싱가포르/타이완 스토어 앱만 지정이 가능했습니다.
{% endhint %}

{% hint style="danger" %}
**주의**: 배포 버전의 바이너리에서는 이 옵션을 반드시 제거해주세요.
{% endhint %}





## 앱에서 원스토어 인앱 결제 라이브러리 적용하기 <a href="#undefined-4" id="undefined-4"></a>

### 로그 레벨 설정 <a href="#undefined-5" id="undefined-5"></a>

개발 단계에서 로그 레벨을 설정하여 SDK의 데이터 흐름을 좀 더 자세히 노출할 수 있습니다. [`android.util.Log`](https://developer.android.com/reference/android/util/Log#summary)에 정의된 값을 기반으로 동작합니다.

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

class _HomePageState extends State<HomePage> {

  @override
  void initState() {
    super.initState();
    // 앱 개발 시 필요에 의해 SDK & Plugin의 로그 레벨을 변경하면 좀 더 자세한 정보를 얻을 수 있습니다.
    // WARNING! Release Build 시엔 로그 레벨 세팅을 제거 바랍니다. (default: Level.info)
    OneStoreLogger.setLogLevel(LogLevel.verbose);
  }
}
```

| 상수             | 값 |
| -------------- | - |
| VERBOSE        | 2 |
| DEBUG          | 3 |
| INFO (default) | 4 |
| WARN           | 5 |
| ERROR          | 6 |

배포 빌드 버전에서는 보안에 취약할 수 있으니 이 옵션을 **삭제**해야 합니다.

### 로그인 요청하기 <a href="#undefined-6" id="undefined-6"></a>

원스토어 인앱 결제는 로그인 기반으로 구동되는 서비스입니다. 앱 최초 시작 시 구매 라이브러리의 API 호출하기 전에 로그인을 유도합니다. 구매 라이브러리 요청 시 토큰 만료나 다른 여러 가지 사항을 미연에 방지할 수 있습니다.

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

final OneStoreAuthClient _authClient = OneStoreAuthClient();

Future<void> launchSignInFlow() async {
  await _authClient.launchSignInFlow().then((signInResult) {
    if (signInResult.isSuccess()) {
      // success
    } else {
      // failure
    }
  });
}
```

### 구매 라이브러리 초기화 <a href="#undefined-7" id="undefined-7"></a>

`PurchaseClientManager` 인스턴스 초기화를 요청합니다. 이때 필요한 값은 public license key입니다. 이 라이선스 키는 원스토어 개발자 센터에 앱 등록을 완료하면 발급받을 수 있습니다.

![](https://onestore-devs-organization.gitbook.io/~gitbook/image?url=https%3A%2F%2F4053804225-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FvkeBBQAZkfKgt5OwYUGz%252Fuploads%252FGObQR8HoAsqSHCiFEssv%252Fimage.png%3Falt%3Dmedia%26token%3D4f12b345-be5e-4fc5-93ed-0ee5649f4b74\&width=768\&dpr=4\&quality=100\&sign=664b42fa\&sv=2)Copy

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

final PurchaseClientManager _clientManager = PurchaseClientManager.instance;
_clientManager.initialize("your license key");
```

public license key는 SDK 내에서 구매 응답에 대한 위변조 체크에 사용됩니다.

### 구매 데이터 업데이트 및 오류 응답 청취하기 <a href="#undefined-8" id="undefined-8"></a>

`PurchaseClientManager.purchasesUpdatedStream` 을 통해 구매 완료 응답을 받을 준비를 합니다.

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

class MyPurchaseManager {
  final PurchaseClientManager _clientManager = PurchaseClientManager.instance;
  late StreamSubscription<List<PurchaseData>> _purchaseDataStream;
  
  MyPurchaseManager() {
    _clientManager.initialize("your license key");
    
    // 구매 완료 후 Stream을 통해 데이터가 전달됩니다.
    _purchaseDataStream = _clientManager.purchasesUpdatedStream.listen(
        (List<PurchaseData> purchasesList) {
      _listenToPurchasesUpdated(purchasesList);
    }, onError: (error) {
      // 구매가 실패 되었거나 유저가 취소가 되었을 때 응답 됩니다.
      _logger.d('purchaseStream error: $error');
    }, onDone: () {
      _purchaseDataStream.cancel();
    });
  }
  

  void _listenToPurchasesUpdated(List<PurchaseData> purchasesList) {
    // do something
  }

}
```

### 상품 상세 정보 조회하기 <a href="#undefined-9" id="undefined-9"></a>

`PurchaseClientManager.queryProductDetails` API를 통해 원스토어 개발자 센터에 등록된 인앱 상품의 상세 정보를 조회할 수 있습니다.

상품 상세 정보는 [`ProductDetail`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/classes/productdetail) 객체를 담은 리스트로 전달됩니다.

| Parameter   | Type                                                                                                                   | Description |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- | ----------- |
| productIds  | List\<String>                                                                                                          | 상품 아이디 리스트 |
| productType | [`ProductType`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype) | 상품 타입       |

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

class MyPurchaseManager {
  final PurchaseClientManager _clientManager = PurchaseClientManager.instance;
  
  static const consumableIds = ['product_1', 'product_2'];
  static const subscriptionIds = ['week', 'month', 'three_month'];
  
  final List<ProductDetail> _products = [];

  Future<void> fetchProductDetails() async {
    var responses = await Future.wait(<Future<ProductDetailsResponse>>[
      _clientManager.queryProductDetails(
        productIds: consumableIds,
        productType: ProductType.inapp
      ),
      _clientManager.queryProductDetails(
        productIds: subscriptionIds,
        productType: ProductType.subs
      )
    ]);
  
    if (responses.first.iapResult.isSuccess()) {
      final List<ProductDetail> result =
          responses.expand((element) => element.productDetailsList).toList();
      _products.clear();
      _products.addAll(result);
      notifyListeners();
    } else {
      _handleError('fetchProductDetails', responses.first.iapResult);
    }
  }
  
}
```

요청하는 상품 리스트가 적다면 위의 예제처럼 상품 타입에 따라 개별 요청이 아닌 [`ProductType.all`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype#id-a-purchaseclient.producttype-all) 타입으로 요청할 수 있습니다.

```dart
class MyPurchaseManager {
  final PurchaseClientManager _clientManager = PurchaseClientManager.instance;
  
  static const consumableIds = ['product_1', 'product_2'];
  static const subscriptionIds = ['week', 'month', 'three_month'];
  
  final List<ProductDetail> _products = [];
  
  Future<void> fetchProductDetails() async {
    var productDetailsResponse = await _clientManager.queryProductDetails(
      productIds: (consumableIds + subscriptionIds),
      productType: ProductType.all
    );
     
    if (productDetailsResponse.iapResult.isSuccess()) {
      final List<ProductDetail> result = productDetailsResponse.productDetailsList;
      _products.clear();
      _products.addAll(result);

    } else {
      _handleError('fetchProductDetails', productDetailsResponse.iapResult);
    }
  }
  
}
```

등록된 인앱상품의 개수가 많은 경우 응답 지연이 발생할 수 있습니다. 이런 경우 안정성이나 속도 측면에서 400개 단위로 호출 하시는 것을 권장 드립니다.

[`ProductType.all`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype#id-a-purchaseclient.producttype-all) 타입은 상품 상세 조회하기에서만 사용할 수 있는 옵션입니다. 다른 API에서는 사용할 수 없습니다.

### 구매 요청하기 <a href="#undefined-10" id="undefined-10"></a>

`PurchaseClientManager.launchPurchaseFlow()` API를 사용하여 구매 요청합니다.

| Parameter        | Type                                                                                                    | Description                                                                                                                                                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| productDetail    | [`ProductDetail`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/classes/productdetail) | [상품 상세 정보 조회하기](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-9) API를 통해 얻는 객체                                                                                                                                               |
| quantity         | Int                                                                                                     | 수량 (default: 1) 복수 구매 시 max: 10                                                                                                                                                                                                                                                         |
| developerPayload | String                                                                                                  | <p>개발사 입력 데이터로 필요한 데이터를 구매 요청 때같이 전송하면 구매 결과(<a href="https://onestore-dev.gitbook.io/dev/v/eng/tools/tools/v21/references/en-classes/en-purchasedata#id-en-purchasedata-getdeveloperpayload"><code>PurchaseData.developerPayload</code></a> )에도 포함되어 전송됨</p><p>제약 사항 : max 200byte</p> |

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

class MyPurchaseManager {
  final PurchaseClientManager _clientManager = PurchaseClientManager.instance;
    
  Future<IapResult> launchPurchaseFlow(ProductDetail productDetail,
      int? quantity, String? developerPayload) async {
    return await _clientManager.launchPurchaseFlow(
        productDetail: productDetail,
        quantity: quantity,
        developerPayload: developerPayload
    );
  }
  
}
```

### 정기 결제 업그레이드 또는 다운그레이드 <a href="#undefined-11" id="undefined-11"></a>

정기 결제는 취소될 때까지 자동으로 갱신됩니다. 정기 결제는 다음 상태를 가질 수 있습니다.

* **활성**: 사용자가 콘텐츠 사용에 문제가 없는 양호한 상태이며 정기 결제에 접근할 수 있습니다.
* **일시 중지 예약**: 사용자가 정기 결제를 이용 중 일시 중지를 하고 싶을 때 선택할 수 있습니다.
  * 주간 정기 결제: 1\~3주 단위로 일시 중지할 수 있습니다.
  * 월간 정기 결제: 1\~3개월 단위로 일시 중지할 수 있습니다.
  * 연간 정기 결제: 일시 중지를 지원하지 않습니다.
* **해지 예약**: 사용자가 정기 결제를 이용 중이지만 취소하고 싶을 때 선택할 수 있습니다. 다음 결제일에 결제가 되지 않습니다.
* **유예, 보류**: 사용자에게 결제 문제가 발생하면 다음 결제일에 결제가 되지 않습니다. 취소 예약을 할 수 없으며 즉시 "구독 해지"를 할 수 있습니다.

정기 결제를 업데이트하기 위해서는 필수로 `비례 배분 모드`를 적용해야 합니다. 아래는 `비례 배분 모드`별 설명입니다.

| Mode                                    | Value | Description                                                                                |
| --------------------------------------- | ----- | ------------------------------------------------------------------------------------------ |
| IMMEDIATE\_WITH\_TIME\_PRORATION        | 1     | 정기 결제의 교체가 즉시 이루어지며, 남은 시간은 가격 차이를 기반으로 조정되어 입금되거나 청구됩니다. (이것은 기본 동작입니다.)                  |
| IMMEDIATE\_AND\_CHARGE\_PRORATED\_PRICE | 2     | 정기 결제의 교체가 즉시 이루어지며, 청구 주기는 동일하게 유지됩니다. 나머지 기간에 대한 가격이 청구됩니다. (이 옵션은 업그레이드에서만 사용할 수 있습니다.) |
| IMMEDIATE\_WITHOUT\_PRORATION           | 3     | 정기 결제의 교체가 즉시 이루어지며, 다음 결제일에 새로운 가격이 청구됩니다. 청구 주기는 동일하게 적용됩니다.                             |
| DEFERRED                                | 4     | 기존 요금제가 만료되면 교체가 적용되며 새 요금이 동시에 청구됩니다.                                                     |

`PurchaseClientManager.launchUpdateSubscription()` API를 통해 요청할 수 있습니다.

정기결제는 [구매 요청하기](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-10)와 동일한 API를 사용하여 사용자에게 업그레이드 또는 다운그레이드를 제공할 수 있습니다. 다만, 정기 결제의 업그레이드 다운그레이드를 적용하기 위해선 기존 정기 결제 구매 토큰과 비례 배분 모드 값이 필수로 필요합니다.

| Parameter       | Type                                                                                                                           | Description                                                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| productDetail   | [`ProductDetail`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/classes/productdetail)                        | [상품 상세 정보 조회하기](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-9) API를 통해 얻는 객체 |
| oldPurchaseData | [`PurchaseData`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/classes/purchasedata)                          | [구매 내역 조회하기](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-13) API를 통해 얻는 객체   |
| prorationMode   | [`ProrationMode`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseflowparams.prorationmode) | 비례 배분 모드                                                                                                                                  |

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

class MyPurchaseManager {
  final PurchaseClientManager _clientManager = PurchaseClientManager.instance;

  Future<IapResult> launchUpdateSubscription(ProductDetail productDetail,
      PurchaseData oldPurchaseData, ProrationMode prorationMode) async {
    return await _clientManager.launchUpdateSubscription(
        productDetail: productDetail,
        oldPurchaseData: oldPurchaseData,
        prorationMode: prorationMode
    );
  }
  
}
```

### 구매 후 처리 <a href="#undefined-12" id="undefined-12"></a>

* `PurchaseClientManager.launchPurchaseFlow()`
* `PurchaseClientManager.launchUpdateSubscription()`

API를 사용하여 구매가 성공적으로 이루어졌다면 [구매 데이터 업데이트 및 오류 응답 청취하기](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-8)서 등록한 `_listenToPurchasesUpdated()` 를 통해 응답을 받을 수 있습니다.

성공적으로 구매 완료되어 응답을 받았다면, 사용자는 소비(Consume) 또는 확인(Acknowledge) 작업을 하는 것이 매우 중요합니다.

3일 이내에 구매를 확인(_acknowledge_) 또는 소비(_consume_)를 하지 않으면 사용자에게 상품이 지급되지 않았다고 판단되어 자동으로 환불됩니다.

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

class MyPurchaseManager {
  final PurchaseClientManager _clientManager = PurchaseClientManager.instance;
  
  final List<ProductDetail> _products = [];
  
  // 상품 상세 정보에서 ProductType.inapp인 것만 필터링 된 데이터
  List<ProductDetail> get consumableProducts => _products
      .where((element) => element.productType == ProductType.inapp)
      .toList();

  // 상품 상세 정보에서 ProductType.subs인 것만 필터링 된 데이터
  List<ProductDetail> get subscriptionProducts => _products
      .where((element) => element.productType == ProductType.subs)
      .toList();

  void _listenToPurchasesUpdated(List<PurchaseData> purchasesList) {
    if (purchasesList.isNotEmpty) {
      for (var element in purchasesList) {
        if (consumableProducts.any((p) => p.productId == element.productId)) {
        /// [ProductType.inapp] 상품은 [consumePurchase] 호출하여 소비합니다.
        } else if (subscriptionProducts.any((p) => p.productId == element.productId)) {
        /// [ProductType.subs] 상품은 [acknowledgePurchase] 호출하여 확인합니다.
        }
      }
    }
  }
  
}
```

#### **소비하기 (Consume)**

`PurchaseClientManager.consumePurchase()` API를 사용하여 관리형 상품([`ProductType.inapp`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype#id-a-purchaseclient.producttype-inapp))을 소비합니다.

관리형 상품([`ProductType.inapp`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype#id-a-purchaseclient.producttype-inapp))은 소비를 하지 않으면 재 구매가 불가합니다.

관리형 상품([`ProductType.inapp`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype#id-a-purchaseclient.producttype-inapp))은 API 사용에 따라 두 가지로 사용될 수 있습니다.

* 소모성 상품: 구매 요청 → 응답 → 아이템 지급 → `consumePurchase`
* 기간제 상품: 구매 요청 → 응답 → 아이템 지급 → `acknowledgePurchase` → 일정 기간이 지난 후 → `consumePurchase`

| Parameter    | Type                                                                                                  | Description                                                                                                                                                                                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| purchaseData | [`PurchaseData`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/classes/purchasedata) | [구매 완료](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-12) 또는 [구매 내역 조회하기](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-13)를 통해 얻는 PurchaseData |

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

class MyPurchaseManager {
  final PurchaseClientManager _clientManager = PurchaseClientManager.instance;

  Future<void> consumePurchase(PurchaseData purchaseData) async {
    await _clientManager
        .consumePurchase(purchaseData: purchaseData)
        .then((iapResult) {
      // IapResult를 통해 해당 API의 성공 여부를 판단할 수 있습니다.
      if (iapResult.isSuccess()) {
        fetchPurchases([ProductType.inapp]);
      }
    });
  }
  
}
```

#### **확인하기 (Acknowledge)**

`PurchaseClientManager.acknowledgePurchase()` API를 사용하여 관리형 상품([`ProductType.inapp`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype#id-a-purchaseclient.producttype-inapp)) 또는 구독형 상품([`ProductType.subs`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype#id-a-purchaseclient.producttype-subs))의 확인을 요청합니다.

구독형 상품([`ProductType.subs`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype#id-a-purchaseclient.producttype-subs))는 소비(Consume)는 할 수 없고 확인(Acknowledge)만 가능합니다.

| Parameter    | Type                                                                                                  | Description                                                                                                                                                                                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| purchaseData | [`PurchaseData`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/classes/purchasedata) | [구매 완료](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-12) 또는 [구매 내역 조회하기](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-13)를 통해 얻는 PurchaseData |

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

class MyPurchaseManager {
  final PurchaseClientManager _clientManager = PurchaseClientManager.instance;

  Future<void> acknowledgePurchase(PurchaseData purchaseData) async {
    await _clientManager
        .acknowledgePurchase(purchaseData: purchaseData)
        .then((iapResult) {
      // IapResult를 통해 해당 API의 성공 여부를 판단할 수 있습니다.
      if (iapResult.isSuccess()) {
        fetchPurchases([ProductType.subs]);
      }
    });
  }
  
}
```

확인 작업이 완료되었다면 구독형 상품([`ProductType.subs`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype#id-a-purchaseclient.producttype-subs))의 경우 [구매 내역 조회하기](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-13)를 통해 [`PurchaseData`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/classes/purchasedata) 업데이트해야 [`PurchaseData.isAcknowledged`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/classes/purchasedata#id-c-purchasedata-isacknowledged) 값이 변경된 것을 확인할 수 있습니다.

### 구매 내역 조회하기 <a href="#undefined-13" id="undefined-13"></a>

`PurchaseClientManager.queryPurchases()` API를 사용하여 소비되지 않은 구매 내역을 요청합니다.

구매 내역 조회하기의 응답 데이터는 [구매 데이터 업데이트 및 오류 응답 청취하기](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-8) 통해 받는 데이터와 동일합니다.

구매 완료 후 데이터를 처리하는 것만으로는 앱이 모든 구매를 처리하는 것을 보장하기에 충분하지 않습니다. 앱에서 사용자가 구매한 모든 항목을 인식하지 못할 수 있습니다. 앱에서 구매 추적을 놓치거나 구매를 인식하지 못할 수 있는 몇 가지 시나리오는 다음과 같습니다.

* **구매 중 네트워크 문제**: 사용자가 구매를 성공적으로 완료하고 원스토어에서 확인을 받았지만 기기가 구매 알림을 받기 전에 네트워크 연결이 끊어졌을 경우
* **여러 기기**: 사용자는 한 기기에서 항목을 구입한 후 기기를 전환할 때 이 항목이 표시되기를 기대합니다.

이러한 상황에 대처하려면 구매 내역 조회하기 API를 상황에 맞게 호출해야 합니다.

* 어플리케이션 처음 구동시
* 어플리케이션이 백그라운드에서 포그라운드로 재 진입했을 경우
* 상점 진입 시

어플리케이션의 상황에 맞게 사용해 주세요.

| Parameter   | Type                                                                                                                   | Description |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- | ----------- |
| productType | [`ProductType`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.producttype) | 상품 타입       |

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

class MyPurchaseManager {
  final PurchaseClientManager _clientManager = PurchaseClientManager.instance;
  
  Future<void> fetchPurchases(ProductType type) async {
    await _clientManager
        .queryPurchases(productType: type)
        .then((response) {
      if (response.iapResult.isSuccess()) {
        if (type == ProductType.inapp) {
          for (var purchaseData in response.purchasesList) {
            consumePurchase(purchaseData);
          }
        } else if (type == ProductType.subs) {
          for (var purchaseData in response.purchasesList) {
            if (!purchaseData.isAcknowledged) {
              acknowledgePurchase(purchaseData);
            }
          }
        }
      } else {
        _handleError('fetchPurchases($type)', response.iapResult);
      }
    });
  }
  
}
```

### 정기 결제 관리 화면 열기 <a href="#undefined-14" id="undefined-14"></a>

`PurchaseClientManager.launchManageSubscription()` API를 사용하여 구독 상품의 상세 페이지로 이동합니다.

구독 상품의 설정 변경은 유저의 몫으로 관리 메뉴에서 할 수 있는 것들은 아래와 같습니다.

* 결제 수단 변경
* 구독 상태 변경 (해지 예약, 취소)
* 기 구독한 상품의 가격 변경 동의

[`PurchaseData`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/classes/purchasedata) 값이 null일 경우 특정 구독 상품의 상세 페이지가 아닌 정기 결제 리스트 화면으로 이동합니다.

| Parameter    | Type                                                                                                  | Description                                                                                                                                                                                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| purchaseData | [`PurchaseData`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/classes/purchasedata) | [구매 완료](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-12) 또는 [구매 내역 조회하기](https://onestore-devs-organization.gitbook.io/guide-inapp-sdk/sdk-purchase-v21/flutter-sdk-v21#undefined-13)를 통해 얻는 PurchaseData |

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

final PurchaseClientManager _clientManager = PurchaseClientManager.instance;
  
Future<void> launchManageSubscription(PurchaseData? purchaseData) async {
  await _clientManager.launchManageSubscription(purchaseData);
}
```



### StoreEnvironment API 기능 추가 <a href="#undefined-15" id="undefined-15"></a>

`StoreEnvironment.getStoreType()` API는 SDK가 탑재된 애플리케이션이 원스토어를 통해 설치되었는지를 판단하는 기능을 제공합니다.

#### &#x20;Store Type 정의

해당 API는 `StoreType`을 반환하며, 아래 네 가지 값 중 하나를 가집니다.

<table><thead><tr><th width="235">StoreType</th><th width="81">value</th><th>description</th></tr></thead><tbody><tr><td><code>StoreType.UNKNOWN</code></td><td>0</td><td>앱 설치 스토어 정보를 알 수 없음 <em>(APK 직접 설치, 출처 불명 등)</em></td></tr><tr><td><code>StoreType.ONESTORE</code></td><td>1</td><td>ONE Store에서 설치됨 <em>(또는 개발자 옵션이 활성화된 경우)</em></td></tr><tr><td>StoreType.VENDING</td><td>2</td><td>Google Play Store에서 설치됨</td></tr><tr><td>StoreType.ETC</td><td>3</td><td>기타 스토어에서 설치됨</td></tr></tbody></table>

#### API 사용 방법

해당 API는 `StoreEnvironment.getStoreType()`을 호출하여 사용할 수 있습니다.

```dart
  StoreType storeType = await OneStoreEnvironment.getStoreType();

  switch (storeType) {
    case StoreType.unknown:
      print("스토어 정보를 알 수 없습니다.");
      break;
    case StoreType.oneStore:
      print("ONE Store에서 설치된 앱입니다.");
      break;
    case StoreType.vending:
      print("Google Play Store에서 설치된 앱입니다.");
      break;
    case StoreType.etc:
      print("기타 스토어에서 설치된 앱입니다.");
      break;
  }
```

**스토어 판단 기준**

이 API는 세 가지 방법을 통해 설치된 스토어를 판별합니다.

1. 원스토어 마켓 서명을 통해 배포된 경우
   1. 원스토어의 마켓 서명을 통한 배포 여부를 확인하여, 원스토어에서 설치된 앱인지 확인합니다.
2. Installer Package Name을 기반으로 판별
   1. 원스토어의 마켓 서명을 통해 배포되지 않은 경우, PackageManager.getInstallerPackageName() API를 이용하여 앱 설치 시 사용된 스토어 정보를 확인합니다.
3. 개발자 옵션(`onestore:dev_option`)이 활성화된 경우
   1. onestore:dev\_option 이 설정 되어있으면 무조건 StoreType.ONESTORE로 응답합니다.

**활용 예시**

스토어별 UI 차별화 적용

원스토어와 다른 앱 마켓에서 제공하는 결제 시스템이 다를 경우, UI를 다르게 설정할 수 있습니다.

```dart
  if (await OneStoreEnvironment.getStoreType() == StoreType.oneStore) {
    showOneStorePaymentUI()
  } else {
    showDefaultPaymentUI()
  }
```

스토어별 기능 차단

특정 기능을 원스토어에서만 사용하도록 설정할 수 있습니다.

```dart
  if (await OneStoreEnvironment.getStoreType() != StoreType.oneStore) {
    print("이 기능은 ONE Store에서만 사용할 수 있습니다.");
  }
  enableOneStoreExclusiveFeature()
```



## 원스토어 서비스 설치하기 <a href="#undefined-15" id="undefined-15"></a>

`PurchaseClientManager.launchUpdateOrInstall()` API를 호출하여 '원스토어 서비스 앱'을 설치할 수 있습니다.

`PurchaseClientManager` API를 사용하는 중에 에러 응답 중 [`RESULT_NEED_UPDATE`](https://onestore-dev.gitbook.io/dev/tools/tools/v21/references/annotations/purchaseclient.responsecode#id-a-purchaseclient.responsecode-result_need_update) 코드가 발생했을 경우가 있습니다. 이는 원스토어 서비스 앱이 설치되지 않았거나 In-app SDK에서 요구하는 버전보다 낮을 경우 발생합니다.

```dart
import 'package:flutter_onestore_inapp/flutter_onestore_inapp.dart';

final PurchaseClientManager _clientManager = PurchaseClientManager.instance;
  
Future<void> launchUpdateOfInstall() async {
  await _clientManager.launchUpdateOrInstall().then((iapResult) {
    if (iapResult.isSuccess()) {
      fetchPurchases();
      fetchProductDetails();
    }
  });
}
```


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/web-payment.md
# 15. 웹 결제 규격 적용하기

## 개요

웹 환경에서 원스토어 인앱결제를 이용하기 위한 연동 방법을 설명합니다.

* [회원 인증](web-payment/member) &#x20;
* [웹에서 원스토어 인앱 결제 사용하기](web-payment/uiapw)

출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/web-payment/member.md
# 회원 인증

## 1.개요 <a href="#id-greater-than-apiv7-greater-than-15.-greater-than-1" id="id-greater-than-apiv7-greater-than-15.-greater-than-1"></a>

Windows 앱을 EXE 유형으로 서비스 하는 경우 웹 결제 규격을 연동해야 합니다.\
웹 결제 규격 연동 시 회원 인증(로그인)이 필요하므로 이 규격서를 제공합니다.

* 결제 서버 API는 회원 인증(로그인) 시 제공되는 OAuth Token(이하 User Access Token)을 이용합니다.
* 결제 시마다 회원 인증(로그인)을 수행하지 않도록 OAuth 인증을 지원합니다.



## 2.주의사항 <a href="#id-greater-than-apiv7-greater-than-15.-greater-than-2" id="id-greater-than-apiv7-greater-than-15.-greater-than-2"></a>

Embedded Browser(웹뷰)에서 구글 SDK를 적용하지 않고, 구글 Oauth 로그인을 시도하는 경우 403 에러가 발생합니다. \
아래의 방법을 통해 403 에러를 방지할 수 있습니다. &#x20;

* Embedded Browser가 아닌 새 창으로 로그인을 시도합니다.&#x20;
* UserAgent 정보를 수정하여 Embedded Browser가 아닌 웹으로 인식하도록 처리합니다.&#x20;



## 3.회원 인증 플로우 <a href="#id-greater-than-apiv7-greater-than-15.-greater-than-3" id="id-greater-than-apiv7-greater-than-15.-greater-than-3"></a>

### 3.1 Conceptual View <a href="#id-greater-than-apiv7-greater-than-15.-greater-than-3" id="id-greater-than-apiv7-greater-than-15.-greater-than-3"></a>

<figure><img src="https://1837360763-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2Fot0z57AnnXZ02C5qyePV%2Fuploads%2FO8FPHSpDj37xXZJyPd1X%2Fimage.png?alt=media&#x26;token=1b516c92-0a38-4187-9b38-a9d4cf5c07c4" alt=""><figcaption></figcaption></figure>

### 3.2 논리 API ↔ 물리 API 규격 매핑 <a href="#id-greater-than-apiv7-greater-than-15.-greater-than-3.2-api-api" id="id-greater-than-apiv7-greater-than-15.-greater-than-3.2-api-api"></a>

Conceptual View'에 표기된 논리 API들이 매핑되는 물리 API 규격은 다음과 같습니다.

<table><thead><tr><th width="81.4444580078125" valign="top">No</th><th width="150.88897705078125" valign="top">Flow</th><th valign="top">논리 API</th><th valign="top">물리 API (연동 규격)</th></tr></thead><tbody><tr><td valign="top">1</td><td valign="top">회원인증</td><td valign="top"><ol><li>Loging UI 요청</li></ol></td><td valign="top">4.2 ONE store 로그인 인증 요청</td></tr><tr><td valign="top">2</td><td valign="top">회원인증</td><td valign="top"><ol start="5"><li>ONEstore UserAccessToken 발급</li></ol></td><td valign="top">4.3 ONE store User Access Token 발급 요청</td></tr></tbody></table>





## 4. ONE store 회원인증 <a href="#id-greater-than-apiv7-greater-than-15.-greater-than-4.onestore" id="id-greater-than-apiv7-greater-than-15.-greater-than-4.onestore"></a>

### 4.1 ONE store 로그인 API 개요 <a href="#id-greater-than-apiv7-greater-than-15.-greater-than-4.1onestore-api" id="id-greater-than-apiv7-greater-than-15.-greater-than-4.1onestore-api"></a>

ONE store 로그인 API는 아래와 같이 구성되어 있습니다.

* ONE store 로그인 인증 요청 API
* 접근 토큰(User Access Token) 발급/삭제 요청 API

ONE store 로그인 인증 요청 API는 개발자 서비스 Web에 ONE store 로그인 화면을 띄우는 API입니다. 사용자가 ONE store 회원 인증에 성공하면 API로부터 받은 임시코드(code) 값을 이용해서 접근 토큰(User Access Token) 발급 요청 API를 호출합니다.

접근 토큰 발급 요청 API를 통해 받은 접근 토큰은 여러가지 Server API들을 호출하는데 사용합니다.

접근 토큰(User Access Token)은 갱신 토큰(Refresh Token)에 비하여 만료기한이 짧으며, 접근 토큰 만료 시 갱신 토큰을 이용하여 새로운 접근 토큰을 발급 받아야 합니다.

토큰 발급 상세 규칙은 '4.1.4 User Access Token 및 Refresh Token 발급 기준'에 명시 되어 있습니다.



#### **4.1.1 표준 응답코드** <a href="#response-code" id="response-code"></a>

<table data-header-hidden><thead><tr><th valign="top"></th><th valign="top"></th><th valign="top"></th><th valign="top"></th><th valign="top"></th><th valign="top"></th></tr></thead><tbody><tr><td valign="top">Code</td><td valign="top">message_ko</td><td valign="top">message_en</td><td valign="top">HTTP Status Code</td><td valign="top">대상API</td><td valign="top">비고</td></tr><tr><td valign="top">Success</td><td valign="top">정상처리 되었습니다.</td><td valign="top">The request has been successfully completed.</td><td valign="top">200 - Success</td><td valign="top">4.4 ONE store User Access Token 삭제 요청<br></td><td valign="top"><br></td></tr><tr><td valign="top">RequiredValueNotExist</td><td valign="top">필수값이 존재하지 않습니다. [ field1, field2, ... ]</td><td valign="top">Request parameters are required. [ field1, field2, ... ]</td><td valign="top">400 - Bad Request</td><td valign="top">공통</td><td valign="top"><br></td></tr><tr><td valign="top">NoSuchData</td><td valign="top">조회된 결과값이 존재하지 않습니다.</td><td valign="top">The requested data could not be found.</td><td valign="top">404 - Not Found</td><td valign="top">단건 조회 API</td><td valign="top"><br></td></tr><tr><td valign="top">ResourceNotFound</td><td valign="top">요청한 자원이 존재하지 않습니다.</td><td valign="top">The requested resource could not be found.</td><td valign="top">404 - Not Found</td><td valign="top">공통</td><td valign="top">요청한 URL 자원이 없는 경우</td></tr><tr><td valign="top">InternalError</td><td valign="top">정의되지 않은 오류가 발생하였습니다.</td><td valign="top">An undefined error has occurred.</td><td valign="top">500 - Internal Server Error</td><td valign="top">공통</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidRequest</td><td valign="top">입력값이 유효하지 않습니다. [ field1, field2, ... ]</td><td valign="top">Request parameters are invalid. [ field1, field2, ... ]</td><td valign="top">400 - Bad Request</td><td valign="top">공통</td><td valign="top"><br></td></tr><tr><td valign="top">UserAccessTokenExpired</td><td valign="top">Access 토큰이 만료되었습니다.</td><td valign="top">User Access Token has expired.</td><td valign="top">401 - Unauthorized</td><td valign="top">공통</td><td valign="top"><p>code 및 UserAccessToken 만료 처리</p><ul><li>만료 기한 초과 시</li><li>회원 상태 변동 시 (휴면/탈퇴/통합)</li><li>TStore ID 회원 패스워드 변경 시</li><li>소셜 인증 실패 시<br>(소셜 연동 정보 변동 - 연결 취소 등)</li></ul></td></tr><tr><td valign="top">InvalidRefreshToken</td><td valign="top">잘못된 Refresh Token 입니다.</td><td valign="top">Invalid refresh token</td><td valign="top">400 - Bad Request</td><td valign="top">공통</td><td valign="top">grant_type : refresh_token 경우</td></tr><tr><td valign="top">ExpiredRefreshToken</td><td valign="top">만료된 Refresh Token 입니다.</td><td valign="top">Invalid refresh token (expired)</td><td valign="top">401 - Unauthorized</td><td valign="top">공통</td><td valign="top">grant_type : refresh_token 경우</td></tr><tr><td valign="top">UnauthorizedAccess</td><td valign="top">해당 API에 접근권한이 없습니다.</td><td valign="top">Not authorized to this API.</td><td valign="top">403 - Fobidden<br></td><td valign="top">공통</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidUserAccessToken</td><td valign="top">Access 토큰이 유효하지 않습니다.</td><td valign="top">User Access Token is invalid.</td><td valign="top">401 - Unauthorized</td><td valign="top">공통</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidAuthorizationParam</td><td valign="top">Authorization Param의 값이 유효하지 않습니다.</td><td valign="top">Authorization param is invalid.</td><td valign="top">400 - Bad Request</td><td valign="top">공통</td><td valign="top"><ul><li>code</li><li>올바르지 않은 code</li></ul></td></tr><tr><td valign="top">MethodNotAllowed</td><td valign="top">지원하지 않는 HTTP Method 입니다.</td><td valign="top">HTTP method not supported.</td><td valign="top">405 - Method Not Allowed</td><td valign="top">공통</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidContentType</td><td valign="top">잘못된 Content Type 입니다.</td><td valign="top">The request content-type is invalid.</td><td valign="top">415 - Unsupported Media Type</td><td valign="top">공통</td><td valign="top"><br></td></tr><tr><td valign="top">UserNotExist</td><td valign="top">회원 정보가 존재하지 않습니다. </td><td valign="top">User does not exist.</td><td valign="top">404 - Not Found</td><td valign="top">공통</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidUser</td><td valign="top">회원 정보가 유효하지 않습니다.</td><td valign="top">User information is not valid.</td><td valign="top">409 - Conflict</td><td valign="top">공통</td><td valign="top"><br></td></tr><tr><td valign="top">UnsupportedResponseType</td><td valign="top"><br></td><td valign="top">Unsupported response types: [field1]</td><td valign="top">400 - Bad Request</td><td valign="top">공통</td><td valign="top"><br></td></tr><tr><td valign="top">WrongApproach</td><td valign="top"><br></td><td valign="top">The wrong approach.</td><td valign="top">403 - Forbidden</td><td valign="top">공통</td><td valign="top"><br></td></tr></tbody></table>

\


#### **4.1.2 표준 오류응답**

오류응답인 경우 표준 응답코드에 정의된 코드 및 메시지를 전달합니다.

**Example**

| <p>HTTP/1.1 400 Bad Request<br>Content-type: application/json;charset=UTF-8<br>{<br>    "error" : {<br>        "code" : "NoSuchData",<br>        "message" : "The requested data could not be found."<br>    }<br>}</p> |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |



#### **4.1.3 client\_id와 client\_secret**

client\_id와 client\_secret 값은 "개발자센터 > 상품현황 > 앱 상세 > 공통정보 > 라이선스 관리" 메뉴를 통해 확인할 수 있습니다. \


#### **4.1.4 User Access Token 및 Refresh Token 발급 기준**

ONE store User Access Token은 Client ID(=게임/앱)와 원스토어 사용자의 회원 ID 기준으로 발급됩니다.&#x20;

* grant\_type : authorization\_code
  * 사용자의 명시적 로그인이기 때문에 기존 token이 유효하지 않아도 신규 발급
  *   user\_access\_token, refresh\_token

      * 기존 유효한 (user\_access, refresh) token이 있는 경우 기존 value 응답
      * 기존 유효한 (user\_access, refresh) token이 없는 경우 신규 발급

      \

* grant\_type : refresh\_token
  * refresh\_token
    * 유효한 refresh\_token이 없는 경우 오류 응답
    * 유효한 refresh\_token이 있는 경우 만료기한 연장(초기화: 기본 35일)
  * user\_access\_token
    * 기존 유효한 user\_access\_token이 있는 경우 기존 value 응답
    * 기존 유효한 user\_access\_token이 없는 경우 신규 발급\
      \

* 토큰 삭제 시 refresh\_token, user\_access\_token 함께 삭제 됨

\


### 4.2 ONE store 로그인 인증 요청 <a href="#id-greater-than-apiv7-greater-than-15.-greater-than-4.2onestore" id="id-greater-than-apiv7-greater-than-15.-greater-than-4.2onestore"></a>

**\[ API Spec. ]**

<table data-header-hidden><thead><tr><th width="229.5555419921875"></th><th></th><th data-hidden></th><th data-hidden></th></tr></thead><tbody><tr><td><strong>Protocol</strong></td><td>HTTPS</td><td></td><td></td></tr><tr><td><strong>Content-Type</strong></td><td>application/x-www-form-urlencoded</td><td></td><td></td></tr><tr><td><strong>Method</strong></td><td>GET/POST</td><td></td><td></td></tr><tr><td><strong>응답 포맷</strong></td><td>URL Redirect</td><td></td><td></td></tr><tr><td><strong>Path</strong></td><td><p>상용</p><p> - <a href="https://accounts.onestore.net/oauth2.0/authorize">https://accounts.onestore.net/oauth2.0/authorize</a></p><p>개발</p><p> - <a href="https://qa-accounts.onestore.co.net/oauth2.0/authorize">https://qa-accounts.onestore.co.net/oauth2.0/authorize</a></p></td><td></td><td></td></tr><tr><td><strong>Description</strong></td><td><p>ONE store OAuth 이용 시 로그인 페이지</p><p>오류 코드 참고 : <a href="#id-4.1.1">표준 응답코드</a></p></td><td></td><td></td></tr></tbody></table>

**\[ Request ]**

**Parameter**

<table data-header-hidden><thead><tr><th valign="top"></th><th width="100.3333740234375" valign="top"></th><th valign="top"></th><th width="109" valign="top"></th><th valign="top"></th></tr></thead><tbody><tr><td valign="top"><strong>Parameter Name</strong></td><td valign="top"><strong>Type</strong></td><td valign="top"><strong>Description</strong></td><td valign="top"><strong>Required</strong></td><td valign="top"><strong>Remarks</strong></td></tr><tr><td valign="top">response_type</td><td valign="top">String</td><td valign="top">인증 과정에 대한 내부 구분값으로 'code'로 전송해야 함</td><td valign="top">Y</td><td valign="top"><ul><li>code</li></ul></td></tr><tr><td valign="top">client_id</td><td valign="top">String</td><td valign="top">OAuth Client Id</td><td valign="top">Y</td><td valign="top">Android OS Application의 Package Name을 사용</td></tr><tr><td valign="top">redirect_uri</td><td valign="top">String</td><td valign="top">등록 시 입력한 Callback URL 값으로 URL 인코딩을 적용한 값</td><td valign="top">Y</td><td valign="top">URL 인코딩</td></tr><tr><td valign="top">state</td><td valign="top">String</td><td valign="top">사이트 간 요청 위조(cross-site request forgery) 공격을 방지하기 위해<br>개발자 측에서 생성한 상태 토큰값으로 URL 인코딩을 적용한 값을 사용</td><td valign="top">Y</td><td valign="top"><ul><li>URL 인코딩</li><li>개발자 측에서 생성/전달. ONE store 측은 넘겨받은 값을 그대로 리턴</li></ul></td></tr><tr><td valign="top">scope</td><td valign="top">String</td><td valign="top">접근 허용 범위를 처리하기 위한 내부 구분값</td><td valign="top">Y</td><td valign="top"><ul><li>user_payment</li></ul></td></tr></tbody></table>

**Header**

<table data-header-hidden><thead><tr><th width="170.4444580078125" valign="top"></th><th width="114.7777099609375" valign="top"></th><th width="307.888916015625" valign="top"></th><th valign="top"></th></tr></thead><tbody><tr><td valign="top"><strong>Parameter Name</strong></td><td valign="top"><strong>Type</strong></td><td valign="top"><strong>Description</strong></td><td valign="top"><strong>Required</strong></td></tr><tr><td valign="top">x-market-code</td><td valign="top">String</td><td valign="top"><p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어(대한민국) </li><li>MKT_GLB: 원스토어(대한민국 외)</li></ul></td><td valign="top">Y</td></tr></tbody></table>

**Example**

| [https://accounts.onestore.net/oauth2.0/authorize?response\_type=code\&client\_id=client\_id\_example\&redirect\_uri=http%3A%2F%2Fservice.redirect.url%2Fsubpath\&state=hLiDdL2uhPtsftcU\&scope=user\_payment](https://accounts.onestore.net/oauth2.0/authorize?response_type=code\&client_id=client_id_example\&redirect_uri=http%3A%2F%2Fservice.redirect.url%2Fsubpath\&state=hLiDdL2uhPtsftcU\&scope=user_payment) |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

**\[ Response ]**

* ONE store 로그인 인증 요청 API를 호출 했을 때 사용자가 로그인되지 않은 상태이면 로그인 화면으로 이동합니다.
* 사용자 로그인 완료 후 약관 및 정보 제공 동의 등이 필요한 경우 해당 화면으로 이동합니다.
* 로그인과 동의 과정이 완료되면 콜백 URL에 code값과 state 값이 URL 문자열로 전송됩니다.
* code 값은 접근 토큰(User Access Token) 발급 요청에 사용됩니다.\
  \


<table data-header-hidden><thead><tr><th width="167.4444580078125" valign="top"></th><th width="108.111083984375" valign="top"></th><th valign="top"></th><th></th></tr></thead><tbody><tr><td valign="top"><strong>Element Name</strong></td><td valign="top"><strong>Type</strong></td><td valign="top"><strong>Description</strong></td><td>Remarks</td></tr><tr><td valign="top">code</td><td valign="top">String</td><td valign="top"><p> - 로그인 인증에 성공하면 반환받는 인증 코드</p><p> - 접근 토큰(User Access Token) 발급에 사용</p></td><td><ul><li>length : 50</li><li>영문 대소문자 및 숫자로 구성</li><li>만료기한 5분</li></ul></td></tr><tr><td valign="top">state<br></td><td valign="top">String</td><td valign="top">사이트 간 요청 위조 공격을 방지하기 위해 개발자 측에서 생성한 상태 토큰</td><td><ul><li>URL 인코딩</li><li>요청 시 넘겨준 값을 그대로 리턴</li></ul></td></tr><tr><td valign="top">error_code</td><td valign="top">String</td><td valign="top">로그인 인증에 실패하면 반환받는 에러 코드</td><td>오류 코드 참고 : <a href="#response-code">표준 응답코드</a></td></tr><tr><td valign="top">error_message</td><td valign="top">String</td><td valign="top"> 로그인 인증에 실패하면 반환받는 에러 메시지</td><td><br></td></tr></tbody></table>

**Example**

| <p>인증 절차 완료 후</p><ul><li>API 요청 성공시 : http://콜백URL?code={code값}&#x26;state={state값}</li><li>API 요청 실패시 : http://콜백URL?state={state값}&#x26;error_code={에러코드값}&#x26;error_message={에러메시지}</li></ul> |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |



**오류 유형 정리**

<table data-header-hidden><thead><tr><th valign="top"></th><th valign="top"></th><th valign="top"></th><th valign="top"></th><th valign="top"></th></tr></thead><tbody><tr><td valign="top">코드</td><td valign="top">응답 방식</td><td valign="top"><strong>발생 조건</strong></td><td valign="top">전문 (메세지 및 화면내용)</td><td valign="top">기타 설명</td></tr><tr><td valign="top">RequiredValueNotExist</td><td valign="top">PAGE</td><td valign="top">필수값 누락<br>- response_type<br>- client_id<br>- state<br>- scope<br>- redirect_uri</td><td valign="top">Request parameters are required. [ {누락 파라미터} ]</td><td valign="top"><br></td></tr><tr><td valign="top">UnsupportedResponseType</td><td valign="top">REDIRECTION</td><td valign="top">파라미터 유효성<br>- response_type ≠ code</td><td valign="top">{redirect_uri}?error_code=UnsupportedResponseType&#x26;error_message=Unsupported response types: [{response_type}]</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidRequest</td><td valign="top">PAGE</td><td valign="top">파라미터 유효성<br>- 미발급 client_id</td><td valign="top">Request parameters are invalid. [ client_id ]</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidRedirect</td><td valign="top">PAGE</td><td valign="top">파라미터 유효성<br>- 미등록 redirect_uri</td><td valign="top">Invalid redirect</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidScope</td><td valign="top">REDIRECTION</td><td valign="top">파라미터 유효성<br>- scope ≠ user_payment</td><td valign="top">{redirect_uri}?error_code=InvalidScope&#x26;error_message=Invalid scope</td><td valign="top">접속 후 인증 요청 시 발생</td></tr><tr><td valign="top">WrongApproach</td><td valign="top">PAGE</td><td valign="top">로그인 URI 직접 접근</td><td valign="top">The wrong approach.</td><td valign="top">/oauth2.0/login 직접 접근</td></tr></tbody></table>

\


### 4.3 ONE store User Access Token 발급 요청 <a href="#id-greater-than-apiv7-greater-than-15.-greater-than-4.3onestoreuseraccesstoken" id="id-greater-than-apiv7-greater-than-15.-greater-than-4.3onestoreuseraccesstoken"></a>

**\[ API Spec. ]**

<table data-header-hidden><thead><tr><th width="217.33331298828125" valign="top"></th><th valign="top"></th><th data-hidden valign="top"></th><th data-hidden valign="top"></th></tr></thead><tbody><tr><td valign="top"><strong>Protocol</strong></td><td valign="top">HTTPS</td><td valign="top"></td><td valign="top"></td></tr><tr><td valign="top"><strong>Content-Type</strong></td><td valign="top">application/x-www-form-urlencoded</td><td valign="top"></td><td valign="top"></td></tr><tr><td valign="top"><strong>Method</strong></td><td valign="top">POST</td><td valign="top"></td><td valign="top"></td></tr><tr><td valign="top"><strong>응답 포맷</strong></td><td valign="top">application/json</td><td valign="top"></td><td valign="top"></td></tr><tr><td valign="top"><strong>Path</strong></td><td valign="top"><p>상용</p><p> - <a href="https://accounts.onestore.net/oauth2.0/token">https://accounts.onestore.net/oauth2.0/token</a></p><p>개발</p><p> - <a href="https://accounts.onestore.net/oauth2.0/token">https://qa-accounts.onestore.net/oauth2.0/token</a></p></td><td valign="top"></td><td valign="top"></td></tr><tr><td valign="top"><strong>Description</strong></td><td valign="top"><ul><li>ONE store User Access Token 발급</li></ul><ul><li>오류 코드 참고 : <a href="#id-4.1.1">표준 응답코드</a></li></ul></td><td valign="top"></td><td valign="top"></td></tr></tbody></table>

**\[ Request ]**

**Parameter**

<table data-header-hidden><thead><tr><th width="161" valign="top"></th><th width="113.77783203125" valign="top"></th><th valign="top"></th><th width="111.22216796875" valign="top"></th><th valign="top"></th></tr></thead><tbody><tr><td valign="top"><strong>Parameter Name</strong></td><td valign="top"><strong>Type</strong></td><td valign="top"><strong>Description</strong></td><td valign="top"><strong>Required</strong></td><td valign="top"><strong>Remarks</strong></td></tr><tr><td valign="top">grant_type</td><td valign="top">String</td><td valign="top">접근 토큰 발급 방식에 대한 구분값</td><td valign="top">Y</td><td valign="top"><ul><li>'authorization_code'</li><li>'refresh_token'</li><li>그 외 발급 방식 현재 미정</li></ul></td></tr><tr><td valign="top">client_id</td><td valign="top">String</td><td valign="top">OAuth Client Id</td><td valign="top">Y</td><td valign="top">Android OS Application의 Package Name을 사용</td></tr><tr><td valign="top">client_secret<br></td><td valign="top">String</td><td valign="top">OAuth Client Secret</td><td valign="top">Y</td><td valign="top">ONE store 개발자센터에서 발급한 Client secret 값을 사용</td></tr><tr><td valign="top">code<br></td><td valign="top">String</td><td valign="top">로그인 인증 요청 API 호출에 성공하고 응답 받은 인증코드값 (code)<br></td><td valign="top">N</td><td valign="top">grant_type : authorization_code 경우 필수값</td></tr><tr><td valign="top">refresh_token</td><td valign="top">String</td><td valign="top">User Access Token 발급 시 함께 발급받은 refresh_token 값</td><td valign="top">N</td><td valign="top">grant_type : refresh_token 경우 필수값</td></tr><tr><td valign="top">state<br></td><td valign="top">String</td><td valign="top">사이트 간 요청 위조(cross-site request forgery) 공격을 방지하기 위해<br>개발자 측에서 생성한 상태 토큰값으로 URL 인코딩을 적용한 값을 사용<br></td><td valign="top">Y</td><td valign="top"><ul><li>URL 인코딩</li><li>개발자 측에서 생성/전달, ONE store 측은 넘겨받은 값을 그대로 리턴 함<br></li></ul></td></tr></tbody></table>

**Header**

<table data-header-hidden><thead><tr><th width="192.33331298828125" valign="top"></th><th width="138.111083984375" valign="top"></th><th valign="top"></th><th width="151.666748046875" valign="top"></th></tr></thead><tbody><tr><td valign="top"><strong>Parameter Name</strong></td><td valign="top"><strong>Type</strong></td><td valign="top"><strong>Description</strong></td><td valign="top"><strong>Required</strong></td></tr><tr><td valign="top">x-market-code</td><td valign="top">String</td><td valign="top"><p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어(대한민국) </li><li>MKT_GLB: 원스토어(대한민국 외)</li></ul></td><td valign="top">Y</td></tr></tbody></table>

**Example**

| <p>POST /oauth2.0/token HTTP/1.1 Host: <a href="http://accounts.onestore.net/">accounts.onestore.net</a> Content-Type: application/x-www-form-urlencoded x-market-code:MKT_ONE</p><p>grant_type : authorization_code의 경우<br>grant_type=authorization_code&#x26;<br>code=EIc5bFrl4RibFls1&#x26;<br>client_id=client_id_example&#x26;<br>client_secret=hDBmMRhz7eJRsM9Z2q1oFBSe&#x26;<br>state=9kgsGTfH4j7IyAkg</p><p>grant_type : refresh_token의 경우<br>grant_type=refresh_token&#x26;<br>refresh_token=EIc5bFrl4RibFls1&#x26;<br>client_id=client_id_example&#x26;<br>client_secret=hDBmMRhz7eJRsM9Z2q1oFBSe&#x26;<br>state=9kgsGTfH4j7IyAkg</p> |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

**\[ Response ]**

<table data-header-hidden><thead><tr><th width="180.5555419921875" valign="top"></th><th width="101.4444580078125" valign="top"></th><th valign="top"></th><th valign="top"></th></tr></thead><tbody><tr><td valign="top"><strong>Element Name</strong></td><td valign="top"><strong>Type</strong></td><td valign="top"><strong>Description</strong></td><td valign="top">Remarks</td></tr><tr><td valign="top">user_access_token<br></td><td valign="top">String</td><td valign="top">ONE store 접근 토큰(User Access Token)</td><td valign="top"><ul><li>max length : 255</li><li>만료기한 10분</li></ul></td></tr><tr><td valign="top">refresh_token</td><td valign="top">String</td><td valign="top">User Access Token에 대한 Refresh 토큰</td><td valign="top"><ul><li>max length : 255</li><li>기본 만료기한 35일, 사용 시 만료기한 연장(초기화)</li></ul></td></tr><tr><td valign="top">token_type</td><td valign="top">String</td><td valign="top">Bearer</td><td valign="top"><br></td></tr><tr><td valign="top">expires_in</td><td valign="top">integer</td><td valign="top">접근 토큰의 유효 기간(초 단위)</td><td valign="top"><br></td></tr><tr><td valign="top">state</td><td valign="top">String</td><td valign="top">사이트 간 요청 위조 공격을 방지하기 위해 개발자 측에서 생성한 상태 토큰</td><td valign="top"><ul><li>URL 인코딩<br></li><li>요청 시 넘겨준 값을 그대로 리턴</li><li>성공 시에만 포함됨</li></ul></td></tr><tr><td valign="top">error {<br></td><td valign="top">Object</td><td valign="top">error 발생 시에만 포함</td><td valign="top">오류 코드 참고 : <a href="#id-4.1.1">표준 응답코드</a></td></tr><tr><td valign="top">code</td><td valign="top">String</td><td valign="top">에러 코드</td><td valign="top"><br></td></tr><tr><td valign="top">message</td><td valign="top">String</td><td valign="top">에러 메세지</td><td valign="top"><br></td></tr><tr><td valign="top">}</td><td valign="top"><br></td><td valign="top"><br></td><td valign="top"><br></td></tr></tbody></table>

**Example**

<table data-header-hidden><thead><tr><th></th></tr></thead><tbody><tr><td><p>// 성공 시</p><p>{</p><pre><code>  "user_access_token" : "f27d2c49-231d-4848-9e8c-ec9a1fef9c35",
  "refresh_token" : "1fe54c5f-60d1-4fbb-a412-929c84adab43",
  "token_type" : "Bearer",
  "expires_in" : 603389,
  "state" : "hLiDdL2uhPtsftcU"
}
</code></pre><p><br></p><p>// 실패 시</p><p><code>{</code><br>    <code>"error" : {</code><br>        <code>"code" : "InvalidAuthorizationParam",</code><br>        <code>"message" : "Authorization param is invalid."</code><br>    <code>}</code><br><code>}</code></p></td></tr></tbody></table>

**오류 유형 정리**

<table data-header-hidden><thead><tr><th valign="top"></th><th valign="top"></th><th valign="top"></th><th valign="top"></th></tr></thead><tbody><tr><td valign="top"><strong>코드</strong></td><td valign="top"><strong>발생 조건</strong></td><td valign="top"><strong>전문 (json)</strong></td><td valign="top"><strong>기타 설명</strong></td></tr><tr><td valign="top">RequiredValueNotExist</td><td valign="top">필수값 누락<br>- grant_type<br>- client_id<br>- client_secret<br>- code (선택적 필수값)<br>- refresh_token (선택적 필수값)<br>- state</td><td valign="top">{<br>    "error": {<br>        "code": "RequiredValueNotExist",<br>        "message": "Request parameters are required. ({누락파라미터})"<br>    }<br>}</td><td valign="top">- code : grant_type : authorization_code 경우 필수값<br>- refresh_token : grant_type : refresh_token 경우 필수값</td></tr><tr><td valign="top">InvalidRequest</td><td valign="top">파라미터 유효성<br>- grant_type ≠ (authorization_code or refresh_token)</td><td valign="top">{<br>    "error": {<br>        "code": "InvalidRequest",<br>        "message": "Request parameters are invalid. [ grant_type ]"<br>    }<br>}</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidAuthorizationParam</td><td valign="top">파라미터 유효성<br>- 미발급 code</td><td valign="top">{<br>    "error": {<br>        "code": "InvalidAuthorizationParam",<br>        "message": "Authorization param is invalid."<br>    }<br>}</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidRequest</td><td valign="top">파라미터 유효성<br>- 미발급 client_id</td><td valign="top">{<br>    "error": {<br>        "code": "InvalidRequest",<br>        "message": "Request parameters are invalid. [ client_id or client_secret ]"<br>    }<br>}</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidRequest</td><td valign="top">파라미터 유효성<br>- 미발급 client_secret</td><td valign="top">{<br>    "error": {<br>        "code": "InvalidRequest",<br>        "message": "Request parameters are invalid. [ client_id or client_secret ]"<br>    }<br>}</td><td valign="top"><br></td></tr><tr><td valign="top">InvalidRefreshToken</td><td valign="top">파라미터 유효성<br>- 미발급 refresh_token</td><td valign="top">{<br>    "error": {<br>        "code": "InvalidRefreshToken",<br>        "message": "Invalid refresh token"<br>    }<br>}</td><td valign="top">grant_type : refresh_token 경우</td></tr><tr><td valign="top">ExpiredRefreshToken</td><td valign="top">파라미터 유효성<br>- 만료 refresh_token</td><td valign="top">{<br>    "error": {<br>        "code": "ExpiredRefreshToken",<br>        "message": "Invalid refresh token (expired)"<br>    }<br>}</td><td valign="top">grant_type : refresh_token 경우</td></tr></tbody></table>


출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/web-payment/uiapw.md

# 웹에서 원스토어 인앱 결제 사용하기

## 개요

웹 환경에서 원스토어 인앱결제를 이용하기 위한 연동 방법을 설명합니다.

## 사전 확인

### 회원인증 연동

윈도우용 게임/앱 또는 PC/Mobile Web 상에서 원스토어 인앱결제를 구현하기 위해서는 회원인증(로그인)연동이 필요합니다.\
자세한 사항은 '[회원인증](member)'을 참고하시기 바랍니다.

### 방화벽 정책 등록

CallBackUrl 호출을 위해 원스토어 결제서버와 개발사 서버간 방화벽 정책 등록이 반드시 필요합니다.

CallBackUrl을 사용하실 개발사는 사전에 서버IP 정보를 전달해 주시기 바랍니다. \
(개발사 서버의 80 혹은 443 포트 연동을 기본으로 하며, 방화벽 정책 등록은 최소 1영업일 이상 소요됩니다.)

## 연동 아키텍쳐

원스토어는 회원 기반 서비스를 제공합니다. 웹 환경에서 결제 시에도 사용자 회원의 로그인이 필요하며, 로그인 이후 결제 요청이 가능합니다.\
결제가 필요한 경우 매번 사용자가 로그인을 수행하지 않도록 OAuth 인증을 지원하며, 모든 구매/결제 관련 서버 API는 로그인 시 제공되는 OAuth Token(이하 User Access Token)을 이용하여 요청해야 합니다.

## 서버 APIs

원스토어 웹 서버 API는 발급받은 접근 토큰(User Access Token)을 이용하여 호출합니다.\
HTTP로 호출 시, Header에 접근 토큰을 전송해 주어야 합니다.

### Standard Response Codes

<table><thead><tr><th width="149">Code</th><th>message_kr</th><th>message_en</th><th>HTTP Status Code</th><th>대상API</th></tr></thead><tbody><tr><td>UserNotExist</td><td>회원 정보가 존재하지 않습니다. </td><td>User does not exist.</td><td>404 - Not Found</td><td>공통</td></tr><tr><td>UserAccessTokenExpired</td><td>User Access Token이 만료되었습니다.</td><td>User Access Token has expired.</td><td>401 - Unauthorized</td><td>공통</td></tr><tr><td>UnsupportedDevice</td><td>상품이 해당 단말을 지원하지 않습니다. </td><td>The product does not support the device.</td><td>400 - Bad Request</td><td>공통</td></tr><tr><td>UnauthorizedUserAccess</td><td>해당 API에 접근권한이 없습니다.</td><td>Not authorized to this API.</td><td>403 - Forbidden<br></td><td>공통</td></tr><tr><td>Success</td><td>정상처리 되었습니다.</td><td>The request has been successfully completed.</td><td>200 - Success</td><td>consume/acknowledge<br></td></tr><tr><td>ServiceMaintenance</td><td>서비스 점검중입니다.</td><td>System maintenance is in progress.</td><td>503 - Service Temporarily Unavailable</td><td>공통</td></tr><tr><td>ResourceNotFound</td><td>요청한 자원이 존재하지 않습니다.</td><td>The requested resource could not be found.</td><td>404 - Not Found</td><td>공통</td></tr><tr><td>RequiredValueNotExist</td><td>필수값이 존재하지 않습니다. [ field1, field2, ... ]</td><td>Request parameters are required. [ field1, field2, ... ]</td><td>400 - Bad Request</td><td>공통</td></tr><tr><td>ProductNotExist</td><td>상품 정보가 존재하지 않습니다.</td><td>The product does not exist.</td><td>404 - Not Found</td><td>공통</td></tr><tr><td>NotSupportMultipleQuantity </td><td>복수 구매 요청은 관리형 상품으로 제한합니다.</td><td>Only Managed products are eligible for repeated purchase requests.</td><td>400 - Bad Request</td><td>requestPurchase</td></tr><tr><td>NoSuchData</td><td>조회된 결과값이 존재하지 않습니다.</td><td>The requested data could not be found.</td><td>404 - Not Found</td><td>단건조회 API</td></tr><tr><td>MethodNotAllowed</td><td>지원하지 않는 HTTP Method 입니다.</td><td>HTTP method not supported.</td><td>405 - Method Not Allowed</td><td>공통</td></tr><tr><td>InvalidUserAccessToken</td><td>User Access Token이 유효하지 않습니다.</td><td>User Access Token is invalid.</td><td>401 - Unauthorized</td><td>공통</td></tr><tr><td>InvalidUser</td><td>회원 정보가 유효하지 않습니다.</td><td>User information is not valid.<br></td><td>409 - Conflict</td><td>공통</td></tr><tr><td>InvalidRequest</td><td>입력값이 유효하지 않습니다. [ field1, field2, ... ]</td><td>Request parameters are invalid. [ field1, field2, ... ]</td><td>400 - Bad Request</td><td>공통</td></tr><tr><td>InvalidPurchaseState</td><td>구매내역이 존재하지 않거나, 구매완료 상태가 아닙니다.</td><td>Purchase history does not exist or is not completed.</td><td>409 - Conflict</td><td>consume/acknowledge<br></td></tr><tr><td>InvalidProduct</td><td>상품 정보가 유효하지 않습니다.</td><td>The product is not valid.</td><td>409 - Conflict</td><td>공통</td></tr><tr><td>InvalidContentType</td><td>잘못된 Content Type 입니다.</td><td>The request content-type is invalid.</td><td>415 - Unsupported Media Type</td><td>공통</td></tr><tr><td>InvalidConsumeState</td><td>소비상태 변경이 불가하거나, 이미 변경완료 되었습니다.</td><td>The purchase consumption status cannot be changed or has already been changed.</td><td>409 - Conflict</td><td>consume</td></tr><tr><td>InvalidAuthorizationHeader</td><td>Authorization 헤더의 값이 유효하지 않습니다.</td><td>Authorization header is invalid.</td><td>400 - Bad Request</td><td>공통</td></tr><tr><td>InternalError</td><td>정의되지 않은 오류가 발생하였습니다.</td><td>An undefined error has occurred.</td><td>500 - Internal Server Error</td><td>공통</td></tr><tr><td>ExceedQuantityMultiplePurchase</td><td>구매 요청이 가능한 개수를 초과하였습니다. (최대 10개)</td><td>Your purchase request has exceeded the quantity available. (Max. 10 items)</td><td>400 - Bad Request</td><td>requestPurchase</td></tr><tr><td>ExceedAmountMultiplePurchase</td><td>구매 요청이 가능한 금액을 초과하였습니다. (최대 50만원)</td><td>Your purchase request has exceeded the amount available. (Max. ₩500,000)</td><td>400 - Bad Request</td><td>requestPurchase</td></tr><tr><td>DeveloperPayloadNotMatch</td><td>구매요청 시 전달된 developerPayload값과 일치하지 않습니다.</td><td>The request developerPayload does not match the value passed in the purchase request.</td><td>400 - Bad Request</td><td>consume/acknowledge</td></tr><tr><td>AlreadyPurchased</td><td>이미 상품을 보유하였거나 함께 구매할 수 없는 상품을 보유중입니다.</td><td>You already have the product or a product that cannot be purchased together.</td><td>409 - Conflict</td><td>requestPurchase</td></tr><tr><td>AccessBlocked</td><td>요청이 차단되었습니다.</td><td>The request was blocked.</td><td>403 - Forbidden</td><td>공통</td></tr></tbody></table>

#### 표준 응답코드

오류응답인 경우 표준 응답코드에 정의된 코드 및 메시지를 전달합니다.

**Example**

```
HTTP/1.1 400 Bad Request
Content-type: application/json;charset=UTF-8
{
    "error" : {
        "code" : "NoSuchData",
        "message" : "The requested data could not be found."
    }
}
```

#### 상품 타입 코드&#x20;

| Code         | Name   | Description              |
| ------------ | ------ | ------------------------ |
| inapp        | 소비성 상품 | consume 가능한 소비성 상품       |
| auto         | 월정액 상품 | 월 자동결제 상품                |
| subscription | 구독형 상품 | 구독형 상품                   |
| all          | 전체 상품  | 소멸성 상품 + 월정액 상품 + 구독형 상품 |

### requestPurchase <a href="#id-2.-apiv7-4.2requestpurchase" id="id-2.-apiv7-4.2requestpurchase"></a>

#### **\[API Spec.]**&#x20;

| **Protocol**     | HTTPS            | **Method** | POST             |
| ---------------- | ---------------- | ---------- | ---------------- |
| **Content-Type** | application/json | 응답 포맷      | application/json |

<table data-header-hidden><thead><tr><th width="121.5859375"></th><th></th></tr></thead><tbody><tr><td><strong>Path</strong></td><td><p>(상용) <a href="https://pcapis.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/%7Btype%7D/products/%7BproductId%7D/order">https://pcapis.onestore.net/pc/v7/apps/{clientId}/purchases/{type}/products/{productId}/order</a></p><p>(개발) <a href="https://sbpp.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/%7Btype%7D/products/%7BproductId%7D/order">https://sbpp.onestore.net/pc/v7/apps/{clientId}/purchases/{type}/products/{productId}/order</a></p></td></tr><tr><td><strong>Description</strong></td><td><p></p><ul><li>특정 인앱상품의 구매를 요청합니다. </li><li>오류 코드 : 표준 응답코드 참조</li></ul></td></tr></tbody></table>

#### **\[ Request ]**

**Parameter**

<table><thead><tr><th>Parameter Name</th><th width="128.93359375">Data Type</th><th width="176.60546875">Required</th><th width="236.91796875">Description</th></tr></thead><tbody><tr><td>clientId</td><td>String</td><td>Y</td><td>API를 호출하는 앱의 클라이언트 ID</td></tr><tr><td>type</td><td>String</td><td>Y</td><td><ul><li>구매를 요청하고자 하는 인앱 상품 타입 코드</li><li>상품 타입 코드 참조</li></ul></td></tr><tr><td>productId</td><td>String</td><td>Y</td><td>구매를 요청하고자 하는 인앱상품 ID</td></tr></tbody></table>

**Header**

| Parameter Name | Data Type | Required | Description                                                                                    |
| -------------- | --------- | -------- | ---------------------------------------------------------------------------------------------- |
| Authorization  | String    | Y        | User Access Token 발급 API를 통해 발급받은 User Access Token                                            |
| x-market-code  | String    | N        | <p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어 (대한민국)         </li><li>MKT_GLB: 원스토어 (대한민국 외) </li></ul> |

**Body**

| Element Name     | Data Type | Data Size   | Required | Description                                                        |
| ---------------- | --------- | ----------- | -------- | ------------------------------------------------------------------ |
| prchsClientPocCd | String    | 50          | Y        | <p>구매 요청 Client 구분 코드<br>POC_PC : PC 결제<br>POC_MOBILE : 모바일 결제</p> |
| returnUrl        | String    | 200         | Y        | 결제결과를 전달받기 위한 redirect URL                                         |
| callbackUrl      | String    | 200         | N        | 결제결과를 전달받기 위한 REST API URL(최종 결제결과만 전달)                            |
| productName      | String    | 50          | N        | 구매를 요청하고자 하는 인앱 상품명, 미 입력시 개발자센터에 등록된 상품명 사용                       |
| developerPayload | String    | 200         | N        | 구매 건을 식별하기 위해 개발사에서 관리하는 식별자                                       |
| quantity         | Integer   | <p><br></p> | N        | 구매하고자 하는 상품의 수량(Default: 1)                                        |

**Example**

```
POST /pc/v7/apps/com.onestorecorp.test/purchases/inapp/products/p5000/order
Host: pcapis.onestore.net Content-Type: application/json
Authorization: Bearer 680b3512-1253-5642-1263-8adjbf651nb
x-market-code: MKT_GLB
 
{
    "prchsClientPocCd": "POC_PC",
    "returnUrl": "https://onestorecorp.com",
    "callbackUrl": "https://onestorecorp.com",
    "productName": "금화100개",
    "developerPayload" : "1jkl2j3lk1lj",
    "quantity": 2
}
```

#### **\[ Response ]**

| Element Name | Data Type | Data Size | Description              |
| ------------ | --------- | --------- | ------------------------ |
| purchaseId   | String    | 20        | 구매 ID                    |
| paymentUrl   | String    | 200       | <p>결제 요청 URL 정보<br></p>  |
| paymentParam | String    | -         | <p>결제 요청 파라미터 정보<br></p> |

**Example**

```
성공 시

{
   "purchaseId": "200406083435101108801",
   "paymentUrl": "https://onestorecorp.com",
   "paymentParam": "ABCDEDIAGJAFERasdfwerewrlkjasjflsdafj42352ds"
}



// 실패 시

{
    "error" : {
        "code" : "AlreadyPurchased",
        "message" : "You already have the product or a product that cannot be purchased together."
    }
}
```

### getProductDetails <a href="#id-2.-apiv7-4.3getproductdetails" id="id-2.-apiv7-4.3getproductdetails"></a>

#### **\[ API Spec. ]**

| **Protocol**     | HTTPS            | **Method**          | POST             |
| ---------------- | ---------------- | ------------------- | ---------------- |
| **Content-Type** | application/json | **Response Format** | application/json |

<table data-header-hidden><thead><tr><th width="123.046875"></th><th></th></tr></thead><tbody><tr><td><strong>Path</strong></td><td><p>(상용) <br><a href="https://pcapis.onestore.net/pc/v7/apps/%7BclientId%7D/products/%7Btype%7D">https://pcapis.onestore.net/pc/v7/apps/{clientId}/products/{type}</a><br></p><p>(개발) <br><a href="https://sbpp.onestore.net/pc/v7/apps/%7BclientId%7D/products/%7Btype%7D">https://sbpp.onestore.net/pc/v7/apps/{clientId}/products/{type}</a></p></td></tr><tr><td><strong>Description</strong></td><td><ul><li>판매 가능한 인앱상품의 상세 정보를 반환합니다.</li><li>오류 코드 : 표준 응답코드 참조</li></ul></td></tr></tbody></table>

#### **\[ Request ]**

**Parameter**

| Parameter Name | Data Type | Required | Description                                                       |
| -------------- | --------- | -------- | ----------------------------------------------------------------- |
|  clientId      | String    | Y        | API를 호출하는 앱의 클라이언트 ID                                             |
| type           | String    | Y        | <ul><li>상품 정보를 조회하고자 하는 인앱 상품 타입 코드</li><li>상품 타입 코드 참조</li></ul> |

**Header**

| Parameter Name | Data Type | Required | Description                                                                                    |
| -------------- | --------- | -------- | ---------------------------------------------------------------------------------------------- |
| Authorization  | String    | Y        | User Access Token 발급 API를 통해 발급받은 User Access Token                                            |
| x-market-code  | String    | N        | <p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어 (대한민국)         </li><li>MKT_GLB: 원스토어 (대한민국 외) </li></ul> |

**Body**

| Element Name     | Data Type   | Data Size   | Required    | Description               |
| ---------------- | ----------- | ----------- | ----------- | ------------------------- |
| productIdList \[ | <p><br></p> | <p><br></p> | <p><br></p> | <p><br></p>               |
| <p><br></p>      | String      | 150         | Y           | 개발자센터에 상품 등록시 지정한 인앱상품 ID |
| ]                | <p><br></p> | <p><br></p> | <p><br></p> | <p><br></p>               |

**Example**

<pre><code>POST /pc/v7/apps/com.onestorecorp.com.test/products/inapp
Host: 
pcapis.onestore.net

Content-Type: application/json
Authorization: Bearer 680b3512-1253-5642-1263-8adjbf651nb
x-market-code: MKT_GLB

{
    "productIdList": ["다이아100_20170818000000", "루비300_20170818000000"]
<strong>}
</strong></code></pre>

#### **\[ Response ]**

| Element Name         | Data Type   | Data Size   | Description                                 |
| -------------------- | ----------- | ----------- | ------------------------------------------- |
| productDetailList \[ | <p><br></p> | -           | 상품 상세 정보 목록                                 |
| {                    | <p><br></p> | <p><br></p> | <p><br></p>                                 |
| productId            | String      | 150         | 구매가능한 상품의 인앱상품 ID                           |
| type                 | String      | 20          | <ul><li>상품 타입</li><li>상품 타입 코드 참조</li></ul> |
| price                | String      | 30          | 상품금액                                        |
| priceCurrencyCode    | String      | 10          | KRW, USD 등의 통화 구분                           |
| title                | String      | -           | 상품명                                         |
| priceAmountMicros    | Long        | -           | 상품금액 \* 100만                                |
| }                    | <p><br></p> | <p><br></p> | <p><br></p>                                 |
| ]                    | <p><br></p> | <p><br></p> | <p><br></p>                                 |

**Example**

```
// 성공 시
{
    "productDetailList": [
        {
            "productId": "다이아100_20170818000000",
            "type": "inapp",
            "price": "1000",
            "priceCurrencyCode": "KRW",
            "title": "Sample Title",
            "priceAmountMicros": 1000000000
        },
        {
            "productId": "루비300_20170818000000",
            "type": "inapp",
            "price": "1000",
            "priceCurrencyCode": "KRW",
            "title": "Sample Title",
            "priceAmountMicros": 1000000000
        }
    ]
}
// 실패 시
{
    "error" : {
        "code" : "InvalidUserAccessToken",
        "message" : "Access token is invalid."
    }
}
```

\


### getPurchases <a href="#id-2.-apiv7-4.4getpurchases" id="id-2.-apiv7-4.4getpurchases"></a>

#### **\[ API Spec. ]**

| **Protocol**     | HTTPS            | **Method** | POST             |
| ---------------- | ---------------- | ---------- | ---------------- |
| **Content-Type** | application/json | 응답 포맷      | application/json |

<table data-header-hidden><thead><tr><th width="143.14453125"></th><th></th></tr></thead><tbody><tr><td><strong>Path</strong></td><td><p>(상용) </p><p><a href="https://pcapis.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/%7Btype%7D">https://pcapis.onestore.net/pc/v7/apps/{clientId}/purchases/{type}</a></p><p></p><p>(개발) <br><a href="https://sbpp.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/%7Btype%7D">https://sbpp.onestore.net/pc/v7/apps/{clientId}/purchases/{type}</a></p></td></tr><tr><td><strong>Description</strong></td><td><ul><li>소비하지 않은 구매목록(수량포함) 반환합니다. (최대 100건까지 조회)</li><li>오류 코드 : 표준 응답코드 참조</li></ul></td></tr></tbody></table>

#### **\[ Request ]**

**Parameter**

| Parameter Name | Data Type | Required | Description                                                                                       |
| -------------- | --------- | -------- | ------------------------------------------------------------------------------------------------- |
| clientId       | String    | Y        | API를 호출하는 앱의 클라이언트 ID                                                                             |
| type           | String    | Y        | <ul><li>상품 정보를 조회하고자 하는 인앱 상품 타입 코드</li><li>상품 타입 코드 참<mark style="color:red;">조</mark></li></ul> |

**Header**

| Parameter Name | Data Type | Required | Description                                                                          |
| -------------- | --------- | -------- | ------------------------------------------------------------------------------------ |
| Authorization  | String    | Y        | User Access Token 발급 API를 통해 발급받은 User Access Token                                  |
| x-market-code  | String    | N        | <p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어 (대한민국)</li><li>MKT_GLB: 원스토어 (대한민국 외)</li></ul> |

**Body**

| Parameter Name  | Data Type | Data Size | Required | Description              |
| --------------- | --------- | --------- | -------- | ------------------------ |
| continuationKey | String    | 41        | N        | 구매 내역 paging 처리를 위한 다음 키 |

**Example**

```
POST /pc/v7/apps/com.onestorecorp.test/purchases/inapp
Host: pcapis.onestore.net Content-Type: application/json
Authorization: Bearer 680b3512-1253-5642-1263-8adjbf651nb
x-market-code: MKT_GLB
 
{
    "continuationKey" : "a2fajklfsjkl2"
}
```

#### **\[ Response ]**

| Element Name             | Data Type         | Data Size   | Description                                                                                 |
| ------------------------ | ----------------- | ----------- | ------------------------------------------------------------------------------------------- |
| productIdList \[         | <p><br></p>       | -           | XVjKVLbw7TIy                                                                                |
| <p><br></p>              | String            | 150         | 개발자센터에 상품 등록 시 지정한 인앱상품 ID                                                                  |
| ]                        | <p><br></p>       | <p><br></p> | <p><br></p>                                                                                 |
| purchaseDetailList \[    | <p><br></p>       | <p><br></p> | <p><br></p>                                                                                 |
| {                        | <p><br></p>       | <p><br></p> | <p><br></p>                                                                                 |
| orderId                  | String            | 40          | 결제ID                                                                                        |
| packageName              | String            | 128         | 구매한 앱의 패키지명                                                                                 |
| productId                | <p>String<br></p> | 150         | <p>개발자센터에 상품 등록 시 지정한 인앱상품 ID<br></p>                                                       |
| purchaseTime             | Long              | -           | 구매 시간                                                                                       |
| acknowledgeState         | Int               | -           | 구매확인 상태( 0: Not Acknowledged, 1: Acknowledged)                                              |
| purchaseState            | Int               | -           | 구매 상태                                                                                       |
| recurringState           | Int               | -           | <p>자동 결제 상태</p><ul><li>0 : 정상가입 상태</li><li>1 : 해지예약 상태</li><li>-1 : 월정액 상품이 아닌 경우</li></ul> |
| purchaseId               | String            | 20          | 구매ID                                                                                        |
| purchaseToken            | String            | 20          | 구매토큰                                                                                        |
| developerPayload         | String            | 200         | 구매건을 식별하기 위해 개발사에서 관리하는 식별자                                                                 |
| quantity                 | Int               | <p><br></p> | 구매 수량                                                                                       |
| }                        | <p><br></p>       | <p><br></p> | <p><br></p>                                                                                 |
| ]                        | <p><br></p>       | <p><br></p> | <p><br></p>                                                                                 |
| purchaseSignatureList \[ | <p><br></p>       | <p><br></p> | <p><br></p>                                                                                 |
| <p><br></p>              | String            | -           | purchaseDetailList 각각의 구매정보 검증을 위한 signature                                                |
| ]                        | <p><br></p>       | <p><br></p> | <p><br></p>                                                                                 |
| continuationKey          | String            | 41          | 구매 내역 paging 처리를 위한 다음 키                                                                    |

**Example**

```
// 성공 시
{
    "productIdList": ["다이아100_20170818000000", "루비300_20170818000000"],
    "purchaseDetailList": [
        {
            "orderId": "01239349082349823489342",
            "packageName": "com.onestore.sample",
            "productId": "다이아100_20170818000000",
            "purchaseTime": 1345678900000,
            "acknowledgeState": 1,
            "purchaseState": 0,
            "recurringState": -1,
            "purchaseId": "17070421461015116878",
            "purchaseToken": "17070421461015116878",
            "developerPayload": "E23DEFB029F84F4383ECB0E53B46B6A2",
            "quantity": 1
        },
        {
            "orderId": "01239349082349823489343",
            "packageName": "com.onestore.sample",
            "productId": "루비300_20170818000000",
            "purchaseTime": 1345678920000,
            "acknowledgeState": 0,
            "purchaseState": 0,
            "recurringState": -1,
            "purchaseId": "17070431461610116878",
            "purchaseToken": "17070421461015116878",
            "developerPayload": "T_RPAY_27_201707120110880",
            "quantity": 2
        }
    ],
    "purchaseSignatureList": ["sign1", "sign2"],
    "continuationKey" : "continuationKey"
}
 
// 실패 시
{
    "error" : {
        "code" : "InvalidUserAccessToken",
        "message" : "Access token is invalid."
    }
}
```

### consumePurchase <a href="#id-2.-apiv7-4.5consumepurchase" id="id-2.-apiv7-4.5consumepurchase"></a>

#### **\[ API Spec. ]**

| **Protocol**     | HTTPS            | **Method** | POST             |
| ---------------- | ---------------- | ---------- | ---------------- |
| **Content-Type** | application/json | 응답 포맷      | application/json |

<table data-header-hidden><thead><tr><th width="129.44921875"></th><th></th></tr></thead><tbody><tr><td><strong>Path</strong></td><td>(상용) <br><a href="https://pcapis.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/inapp/%7BpurchaseToken%7D/consume">https://pcapis.onestore.net/pc/v7/apps/{clientId}/purchases/inapp/{purchaseToken}/consume</a><br><br>(개발) <br><a href="https://sbpp.onestore.co.kr/pc/v7/apps/%7BclientId%7D/purchases/inapp/%7BpurchaseToken%7D/consume">https://sbpp.onestore.co.kr/pc/v7/apps/{clientId}/purchases/inapp/{purchaseToken}/consume</a></td></tr><tr><td><strong>Description</strong></td><td><ul><li>구매한 관리형 인앱 상품을 소비한 상태로 변경합니다. (소비성 상품만 사용 가능)</li><li>오류 코드 : 표준 응답코드 참조</li></ul></td></tr></tbody></table>

#### **\[ Request ]**

**Parameter**

| Parameter Name | Data Type | Required | Description           |
| -------------- | --------- | -------- | --------------------- |
| clientId       | String    | Y        | API를 호출하는 앱의 클라이언트 ID |
| purchaseToken  | String    | Y        | 구매토큰                  |

**Header**

| Parameter Name | Data Type | Required | Description                                                                          |
| -------------- | --------- | -------- | ------------------------------------------------------------------------------------ |
| Authorization  | String    | Y        | User Access Token 발급 API를 통해 발급받은 User Access Token                                  |
| x-market-code  | String    | N        | <p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어 (대한민국)</li><li>MKT_GLB: 원스토어 (대한민국 외)</li></ul> |

**Body**

| Parameter Name   | Data Type | Data Size | Required | Description                  |
| ---------------- | --------- | --------- | -------- | ---------------------------- |
| developerPayload | String    | 200       | N        | 구매 건을 식별하기 위해 개발사에서 관리하는 식별자 |

**Example**

```
POST /pc/v7/apps/com.onestorecorp.test/purchases/inapp/200406083435101108801/consume
Host: http://pcapis.onestore.net
Content-Type: application/json
Authorization: Bearer 680b3512-1253-5642-1263-8adjbf651nb
x-market-code: MKT_GLB

{
"developerPayload" : "1jkl2j3lk1lj"
}
```

#### **\[ Response ]**

| Element Name | Data Type   | Data Size   | Description |
| ------------ | ----------- | ----------- | ----------- |
| result       | Object      | <p><br></p> | <p><br></p> |
| {            | <p><br></p> | <p><br></p> | <p><br></p> |
| code         | String      | 50          | 응답코드(정상처리)  |
| message      | String      | 300         | 응답메시지(정상처리) |
| }            | <p><br></p> | <p><br></p> | <p><br></p> |

**Example**

```
/// 성공 시

{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}



// 실패 시

{
    "error" : {
        "code" : "AlreadyPurchased",
        "message" : "You already have the product or a product that cannot be purchased together."
    }
}
```

### acknowledgePurchase

#### **\[ API Spec. ]**

| **Protocol**     | HTTPS            | **Method** | POST             |
| ---------------- | ---------------- | ---------- | ---------------- |
| **Content-Type** | application/json | 응답 포맷      | application/json |

<table data-header-hidden><thead><tr><th width="125.921875"></th><th></th></tr></thead><tbody><tr><td><strong>Path</strong></td><td><p>(상용) <br><a href="https://pcapis.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/all/%7BpurchaseToken%7D/acknowledge">https://pcapis.onestore.net/pc/v7/apps/{clientId}/purchases/all/{purchaseToken}/acknowledge</a><br></p><p>(개발) <br><a href="https://sbpp.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/all/%7BpurchaseToken%7D/acknowledge">https://sbpp.onestore.net/pc/v7/apps/{clientId}/purchases/all/{purchaseToken}/acknowledge</a></p></td></tr><tr><td><strong>Description</strong></td><td><ul><li>구매한 인앱 상품을 구매확인 상태로 변경합니다. (소멸성 상품, 월정액 상품 모두 지원)</li><li>오류 코드 : 표준 응답코드 참조</li></ul></td></tr></tbody></table>

#### **\[ Request ]**

**Parameter**

| Parameter     | Data Type | Required | Description           |
| ------------- | --------- | -------- | --------------------- |
|  clientId     | String    | Y        | API를 호출하는 앱의 클라이언트 ID |
| purchaseToken | String    | Y        | 구매토큰                  |

**Header**

| Parameter Name | Data Type | Required | Description                                                                          |
| -------------- | --------- | -------- | ------------------------------------------------------------------------------------ |
| Authorization  | String    | Y        | User Access Token 발급 API를 통해 발급받은 User Access Token                                  |
| x-market-code  | String    | N        | <p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어 (대한민국)</li><li>MKT_GLB: 원스토어 (대한민국 외)</li></ul> |

**Body**

<table><thead><tr><th width="157.7734375">Element Name</th><th>Data Type</th><th>Data Size</th><th>Required</th><th>Description</th></tr></thead><tbody><tr><td>developerPayload</td><td>String</td><td>200</td><td>N</td><td>구매 건을 식별하기 위해 개발사에서 관리하는 식별자</td></tr></tbody></table>

**Example**

```
POST /pc/v7/apps/com.onestorecorp.test/purchases/inapp/200406083435101108801/acknowledge
Host: pcapis.onestore.net
Content-Type: application/json
Authorization: Bearer 680b3512-1253-5642-1263-8adjbf651nb
x-market-code: MKT_GLB

{
"developerPayload" : "1jkl2j3lk1lj"
}
```

#### **\[ Response ]**

| Element Name | Data Type   | Data Size   | Description |
| ------------ | ----------- | ----------- | ----------- |
| result       | Object      | <p><br></p> | <p><br></p> |
| {            | <p><br></p> | <p><br></p> | <p><br></p> |
| code         | String      | 50          | 응답코드(정상처리)  |
| message      | String      | 300         | 응답메시지(정상처리) |
| }            | <p><br></p> | <p><br></p> | <p><br></p> |

**Example**

```
// 성공 시

{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}



// 실패 시

{
    "error" : {
        "code" : "AlreadyPurchased",
        "message" : "You already have the product or a product that cannot be purchased together."
    }
}
```

### cancelRecurringPurchase <a href="#id-2.-apiv7-4.7cancelrecurringpurchase" id="id-2.-apiv7-4.7cancelrecurringpurchase"></a>

#### **\[ API Spec. ]**

| **Protocol**     | HTTPS            | **Method** | POST             |
| ---------------- | ---------------- | ---------- | ---------------- |
| **Content-Type** | application/json | 응답 포맷      | application/json |

<table data-header-hidden><thead><tr><th width="121.58984375"></th><th></th></tr></thead><tbody><tr><td><strong>Path</strong></td><td><p>(상용) <br><a href="https://pcapis.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/auto/%7BpurchaseToken%7D/cancel">https://pcapis.onestore.net/pc/v7/apps/{clientId}/purchases/auto/{purchaseToken}/cancel</a><br></p><p>(개발) <br><a href="https://sbpp.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/auto/%7BpurchaseToken%7D/cancel">https://sbpp.onestore.net/pc/v7/apps/{clientId}/purchases/auto/{purchaseToken}/cancel</a></p></td></tr><tr><td><strong>Description</strong></td><td><ul><li>월정액(자동결제) 상품의 다음 자동결제를 취소 예약(해지 예약)합니다.</li><li>오류 코드 : 표준 응답코드 참조</li></ul></td></tr></tbody></table>

#### **\[ Request ]**

**Parameter**

| Parameter Name | Data Type | Required | Description           |
| -------------- | --------- | -------- | --------------------- |
|  clientId      | String    | Y        | API를 호출하는 앱의 클라이언트 ID |
| purchaseToken  | String    | Y        | 구매토큰                  |

**Header**

| Parameter Name | Data Type | Required | Description                                                                          |
| -------------- | --------- | -------- | ------------------------------------------------------------------------------------ |
| Authorization  | String    | Y        | User Access Token 발급 API를 통해 발급받은 User Access Token                                  |
| x-market-code  | String    | N        | <p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어 (대한민국)</li><li>MKT_GLB: 원스토어 (대한민국 외)</li></ul> |

**Body**

N/A

**Example**

```
POST /pc/v7/apps/com.onestorecorp.test/purchases/auto/200406083435101108801/cancel
Host: pcapis.onestore.net
Content-Type: application/json
Authorization: Bearer 680b3512-1253-5642-1263-8adjbf651nb
x-market-code: MKT_GLB

{
}
```

#### **\[ Response ]**

| Element Name | Data Type   | Data Size   | Description |
| ------------ | ----------- | ----------- | ----------- |
| result       | Object      | <p><br></p> | <p><br></p> |
| {            | <p><br></p> | <p><br></p> | <p><br></p> |
| code         | String      | 50          | 응답코드(정상처리)  |
| message      | String      | 300         | 응답메시지(정상처리) |
| }            | <p><br></p> | <p><br></p> | <p><br></p> |

**Example**

```
// 성공 시

{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}



// 실패 시

{
    "error" : {
        "code" : "NoSuchData",
        "message" : "The requested data could not be found."
    }
}
```

### reactivateRecurringPurchase <a href="#id-2.-apiv7-4.8reactivaterecurringpurchase" id="id-2.-apiv7-4.8reactivaterecurringpurchase"></a>

#### **\[ API Spec. ]**

| **Protocol**     | HTTPS            | **Method** | POST             |
| ---------------- | ---------------- | ---------- | ---------------- |
| **Content-Type** | application/json | 응답 포맷      | application/json |

<table data-header-hidden><thead><tr><th width="126.37890625"></th><th></th></tr></thead><tbody><tr><td><strong>Path</strong></td><td><p>(상용) <a href="https://pcapis.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/auto/%7BpurchaseToken%7D/reactivate">https://pcapis.onestore.net/pc/v7/apps/{clientId}/purchases/auto/{purchaseToken}/reactivate</a><br></p><p>(개발) <br><a href="https://sbpp.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/auto/%7BpurchaseToken%7D/reactivate">https://sbpp.onestore.net/pc/v7/apps/{clientId}/purchases/auto/{purchaseToken}/reactivate</a></p></td></tr><tr><td><strong>Description</strong></td><td><ul><li>월정액(자동결제) 상품의 기존 취소 예약(해지 예약)을 취소하여 다음 자동결제가 정상적으로 진행되도록 합니다.</li><li>이 API는 요청하는 월정액(자동결제) 상품의 상태가 취소 예약 상태일 때만 정상동작합니다.</li><li>오류 코드 : 표준 응답코드 참조</li></ul></td></tr></tbody></table>

#### **\[ Request ]**

**Parameter**

| Parameter Name | Data Type | Required | Description           |
| -------------- | --------- | -------- | --------------------- |
| clientId       | String    | Y        | API를 호출하는 앱의 클라이언트 ID |
| purchaseToken  | String    | Y        | 구매토큰                  |

**Header**

| Parameter Name | Data Type | Required | Description                                                                          |
| -------------- | --------- | -------- | ------------------------------------------------------------------------------------ |
| Authorization  | String    | Y        | User Access Token 발급 API를 통해 발급받은 User Access Token                                  |
| x-market-code  | String    | N        | <p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어 (대한민국)</li><li>MKT_GLB: 원스토어 (대한민국 외)</li></ul> |

**Body**

N/A

**Example**

```
POST /pc/v7/apps/com.onestorecorp.test/purchases/auto/200406083435101108801/reactivate
Host: pcapis.onestore.net
Content-Type: application/json
Authorization: Bearer 680b3512-1253-5642-1263-8adjbf651nb
x-market-code: MKT_GLB

{
}
```

#### **\[ Response ]**

| Element Name | Data Type   | Data Size   | Description |
| ------------ | ----------- | ----------- | ----------- |
| result       | Object      | <p><br></p> | <p><br></p> |
| {            | <p><br></p> | <p><br></p> | <p><br></p> |
| code         | String      | 50          | 응답코드(정상처리)  |
| message      | String      | 300         | 응답메시지(정상처리) |
| }            | <p><br></p> | <p><br></p> | <p><br></p> |

**Example**

```
// 성공 시

{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}



// 실패 시

{
    "error" : {
        "code" : "NoSuchData",
        "message" : "The requested data could not be found."
    }
}
```

### cancelSubscription <a href="#id-2.-apiv7-4.9cancelsubscription" id="id-2.-apiv7-4.9cancelsubscription"></a>

#### **\[ API Spec. ]**

| **Protocol**     | HTTPS            | **Method** | POST             |
| ---------------- | ---------------- | ---------- | ---------------- |
| **Content-Type** | application/json | 응답 포맷      | application/json |

<table data-header-hidden><thead><tr><th width="127.9375"></th><th></th></tr></thead><tbody><tr><td><strong>Path</strong></td><td><p>(상용) <br><a href="https://pcapis.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/subscription/%7BpurchaseToken%7D/cancel">https://pcapis.onestore.net/pc/v7/apps/{clientId}/purchases/subscription/{purchaseToken}/cancel</a><br></p><p>(개발) <br><a href="https://sbpp.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/subscription/%7BpurchaseToken%7D/cancel">https://sbpp.onestore.net/pc/v7/apps/{clientId}/purchases/subscription/{purchaseToken}/cancel</a></p></td></tr><tr><td><strong>Description</strong></td><td><ul><li>구독형 상품의 다음 자동결제를 취소 예약(해지 예약)합니다.</li><li>오류 코드 : 표준 응답코드 참조</li></ul></td></tr></tbody></table>

#### **\[ Request ]**

**Parameter**

| Parameter Name | Data Type | Required | Description           |
| -------------- | --------- | -------- | --------------------- |
| clientId       | String    | Y        | API를 호출하는 앱의 클라이언트 ID |
| purchaseToken  | String    | Y        | 구매토큰                  |

**Header**

| Parameter Name | Data Type | Required | Description                                                                          |
| -------------- | --------- | -------- | ------------------------------------------------------------------------------------ |
| Authorization  | String    | Y        | User Access Token 발급 API를 통해 발급받은 User Access Token                                  |
| x-market-code  | String    | N        | <p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어 (대한민국)</li><li>MKT_GLB: 원스토어 (대한민국 외)</li></ul> |

**Body**

N/A

**Example**

```
POST /pc/v7/apps/com.onestorecorp.test/purchases/subscription/200406083435101108801/cancel
Host: pcapis.onestore.net
Content-Type: application/json
Authorization: Bearer 680b3512-1253-5642-1263-8adjbf651nb
x-market-code: MKT_GLB
 
{  
}
```

#### **\[ Response ]**

| Element Name | Data Type   | Data Size   | Description |
| ------------ | ----------- | ----------- | ----------- |
| result       | Object      | <p><br></p> | <p><br></p> |
| {            | <p><br></p> | <p><br></p> | <p><br></p> |
| code         | String      | 50          | 응답코드(정상처리)  |
| message      | String      | 300         | 응답메시지(정상처리) |
| }            | <p><br></p> | <p><br></p> | <p><br></p> |

**Example**

```
// 성공 시
{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}
 
 
// 실패 시
{
    "error" : {
        "code" : "NoSuchData",
        "message" : "The requested data could not be found."
    }
}
```

### reactivateSubscription <a href="#id-2.-apiv7-4.10reactivatesubscription" id="id-2.-apiv7-4.10reactivatesubscription"></a>

#### **\[ API Spec. ]**

| **Protocol**     | HTTPS            | **Method** | POST             |
| ---------------- | ---------------- | ---------- | ---------------- |
| **Content-Type** | application/json | 응답 포맷      | application/json |

<table data-header-hidden><thead><tr><th width="129.203125"></th><th></th></tr></thead><tbody><tr><td><strong>Path</strong></td><td><p>(상용) <br><a href="https://pcapis.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/subscription/%7BpurchaseToken%7D/reactivate">https://pcapis.onestore.net/pc/v7/apps/{clientId}/purchases/subscription/{purchaseToken}/reactivate</a><br></p><p>(개발) <br><a href="https://sbpp.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/subscription/%7BpurchaseToken%7D/reactivate">https://sbpp.onestore.net/pc/v7/apps/{clientId}/purchases/subscription/{purchaseToken}/reactivate</a></p></td></tr><tr><td><strong>Description</strong></td><td><ul><li>구독형 상품의 다음 자동결제를 취소 예약(해지 예약)을 취소합니다.</li><li>오류 코드 : 표준 응답코드 참조</li></ul></td></tr></tbody></table>

#### **\[ Request ]**

**Parameter**

| Parameter Name | Data Type | Required | Description           |
| -------------- | --------- | -------- | --------------------- |
| clientId       | String    | Y        | API를 호출하는 앱의 클라이언트 ID |
| purchaseToken  | String    | Y        | 구매토큰                  |

**Header**

| Parameter Name | Data Type | Required | Description                                                                          |
| -------------- | --------- | -------- | ------------------------------------------------------------------------------------ |
| Authorization  | String    | Y        | User Access Token 발급 API를 통해 발급받은 User Access Token                                  |
| x-market-code  | String    | N        | <p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어 (대한민국)</li><li>MKT_GLB: 원스토어 (대한민국 외)</li></ul> |

**Body**

**Example**

```
POST /pc/v7/apps/com.onestorecorp.test/purchases/subscription/200406083435101108801/reactivate
Host: pcapis.onestore.net 
Content-Type: application/json
Authorization: Bearer 680b3512-1253-5642-1263-8adjbf651nb
x-market-code: MKT_GLB
 
{
}

```

#### **\[ Response ]**

| Element Name | Data Type   | Data Size   | Description |
| ------------ | ----------- | ----------- | ----------- |
| result       | Object      | <p><br></p> | <p><br></p> |
| {            | <p><br></p> | <p><br></p> | <p><br></p> |
| code         | String      | 50          | 응답코드(정상처리)  |
| message      | String      | 300         | 응답메시지(정상처리) |
| }            | <p><br></p> | <p><br></p> | <p><br></p> |

**Example**

```
// 성공 시
{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    }
}
 
 
// 실패 시
{
    "error" : {
        "code" : "NoSuchData",
        "message" : "The requested data could not be found."
    }
}
```

### getSubscriptionDetail <a href="#id-2.-apiv7-4.11getsubscriptiondetail" id="id-2.-apiv7-4.11getsubscriptiondetail"></a>

#### **\[ API Spec. ]**

| **Protocol**     | HTTPS            | **Method** | POST             |
| ---------------- | ---------------- | ---------- | ---------------- |
| **Content-Type** | application/json | 응답 포맷      | application/json |

<table data-header-hidden><thead><tr><th width="128.640625"></th><th></th></tr></thead><tbody><tr><td><strong>Path</strong></td><td><p>(상용) <br><a href="https://pcapis.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/subscription/%7BpurchaseToken%7D">https://pcapis.onestore.net/pc/v7/apps/{clientId}/purchases/subscription/{purchaseToken}</a><br></p><p>(개발) <br><a href="https://sbpp.onestore.net/pc/v7/apps/%7BclientId%7D/purchases/subscription/%7BpurchaseToken%7D">https://sbpp.onestore.net/pc/v7/apps/{clientId}/purchases/subscription/{purchaseToken}</a></p></td></tr><tr><td><strong>Description</strong></td><td><ul><li>구독의 상세정보를 조회한다.</li><li>오류 코드 : 표준 응답코드 참조</li></ul></td></tr></tbody></table>

#### **\[ Request ]**

**Path Parameter**

| Parameter Name | Data Type | Required | Description           |
| -------------- | --------- | -------- | --------------------- |
| clientId       | String    | Y        | API를 호출하는 앱의 클라이언트 ID |
| purchaseToken  | String    | Y        | 구매토큰                  |

**Header**

| Parameter Name | Data Type | Required | Description                                                                          |
| -------------- | --------- | -------- | ------------------------------------------------------------------------------------ |
| Authorization  | String    | Y        | User Access Token 발급 API를 통해 발급받은 User Access Token                                  |
| x-market-code  | String    | N        | <p>마켓 구분 코드</p><ul><li>MKT_ONE: 원스토어 (대한민국)</li><li>MKT_GLB: 원스토어 (대한민국 외)</li></ul> |

**Body**

**Example**

```
POST /pc/v7/apps/com.onestorecorp.test/purchases/subscription/200406083435101108801
Host: pcapis.onestore.net Content-Type: application/json
Authorization: Bearer 680b3512-1253-5642-1263-8adjbf651nb
x-market-code: MKT_GLB
 
{
}
```

#### **\[ Response ]**

<table><thead><tr><th width="149">Element Name</th><th></th><th></th><th>Data Type</th><th>Description</th></tr></thead><tbody><tr><td>result {</td><td></td><td></td><td>Object</td><td>API호출 결과(정상) - 정상 시에 응답</td></tr><tr><td></td><td>code</td><td></td><td>String</td><td>응답코드</td></tr><tr><td></td><td>message</td><td></td><td>String</td><td>응답메시지</td></tr><tr><td>}</td><td><br></td><td><br></td><td></td><td></td></tr><tr><td>error {</td><td></td><td></td><td>Object</td><td>API호출 결과(에러) - 에러 시에 응답</td></tr><tr><td></td><td>code</td><td></td><td>String</td><td>응답코드</td></tr><tr><td></td><td>message</td><td></td><td>String</td><td>응답메시지</td></tr><tr><td>}</td><td><br></td><td><br></td><td></td><td></td></tr><tr><td>subscription {</td><td></td><td><br></td><td>Object</td><td></td></tr><tr><td><br></td><td>productId</td><td></td><td>String</td><td>커스텀 상품 ID</td></tr><tr><td><br></td><td>productName</td><td></td><td>String</td><td>상품명</td></tr><tr><td><br></td><td>parentProductId</td><td></td><td>String</td><td>모상품 ID</td></tr><tr><td><br></td><td>parentProductName</td><td></td><td>String</td><td>모상품명</td></tr><tr><td><br></td><td>packageName</td><td></td><td>String</td><td>패키지명</td></tr><tr><td><br></td><td>productAmount</td><td></td><td>String</td><td>상품 금액</td></tr><tr><td><br></td><td>productAmountMicros</td><td></td><td>Long</td><td>상품 금액 * 100만</td></tr><tr><td><br></td><td>priceCurrencyCode<br></td><td></td><td>String</td><td>KRW, USD 등등의 통화 구분</td></tr><tr><td><br></td><td>imagePath</td><td></td><td>String</td><td>상품 이미지 경로</td></tr><tr><td><br></td><td>periodUnit</td><td></td><td>String</td><td>이용 기간 단위</td></tr><tr><td><br></td><td>period</td><td></td><td>Integer</td><td>이용 기간</td></tr><tr><td><br></td><td>purchaseToken</td><td></td><td>String</td><td>구매 토큰</td></tr><tr><td><br></td><td>status</td><td></td><td>String</td><td>구독 상태 코드</td></tr><tr><td><br></td><td>startDate</td><td></td><td>Long</td><td>구독 시작(첫 결제) 일시(millis)</td></tr><tr><td><br></td><td>expiryDate</td><td></td><td>Long</td><td>구독 만료 일시(millis)</td></tr><tr><td><br></td><td>paymentAmount</td><td></td><td>String</td><td>이전 결제 금액</td></tr><tr><td><br></td><td>paymentAmountMicros</td><td></td><td>Long</td><td>이전 결제 금액 * 100만</td></tr><tr><td><br></td><td>nextPaymentAmount</td><td></td><td>String</td><td>다음 결제 금액 </td></tr><tr><td><br></td><td>nextPaymentAmountMicros</td><td></td><td>Long</td><td>다음 결제 금액 * 100만</td></tr><tr><td><br></td><td>nextPaymentDate</td><td></td><td>Long</td><td>다음 결제 일시(millis)</td></tr><tr><td><br></td><td>pauseAllow</td><td></td><td>String</td><td>일시중지 사용 가능 여부(Y/N(default))</td></tr><tr><td><br></td><td>pauseStartDate</td><td></td><td>Long</td><td>일시중지 시작일시(millis) - 일시중지예약/일시중지 상태일 경우에만 응답</td></tr><tr><td><br></td><td>pauseEndDate</td><td></td><td>Long</td><td>일시중지 종료일시(millis) - 일시중지예약/일시중지 상태일 경우에만 응답 </td></tr><tr><td><br></td><td>promotionAmount</td><td></td><td>String</td><td>프로모션 상품 금액</td></tr><tr><td><br></td><td>promotionAmountMicros</td><td></td><td>Long</td><td>프로모션 상품 금액 * 100만</td></tr><tr><td><br></td><td>promotionPeriod</td><td></td><td>Integer</td><td>프로모션 이용 기간</td></tr><tr><td><br></td><td>priceChanges [</td><td></td><td>Array</td><td>가격 변동 정보 목록</td></tr><tr><td><br></td><td>{</td><td><br></td><td><br></td><td><br></td></tr><tr><td><br></td><td><br></td><td>priceChangeSeq</td><td>Integer</td><td>가격 변동 시퀀스 </td></tr><tr><td><br></td><td><br></td><td>priceChangeApplyStartDate</td><td>Long</td><td>가격 변동 적용 시작 시간</td></tr><tr><td><br></td><td><br></td><td>priceChangePreviousAmount</td><td>String</td><td>가격 변동 이전 금액</td></tr><tr><td><br></td><td><br></td><td>priceChangePreviousAmountMicros</td><td>Long</td><td>가격 변동 이전 금액 * 100만</td></tr><tr><td><br></td><td><br></td><td>priceChangeAmount</td><td>String</td><td>가격 변동 금액</td></tr><tr><td><br></td><td><br></td><td>priceChangeAmountMicros</td><td>Long</td><td>가격 변동 금액 * 100만</td></tr><tr><td><br></td><td><br></td><td>priceChangeAgreement</td><td>String</td><td>가격 변동 동의 여부</td></tr><tr><td><br></td><td><br></td><td>priceChangeAgreementDueDate</td><td>Long</td><td><p>가격 변동 동의 만료 시간</p><p>(정책 부연 설명)</p><ul><li>Value = 가격변경일 +7+30일</li><li>사용자는 동의만료일 이후 첫 자동결제 시점까지 동의가능함.</li></ul></td></tr><tr><td><br></td><td>}]</td><td><br></td><td><br></td><td><br></td></tr><tr><td>}</td><td><br></td><td><br></td><td></td><td></td></tr></tbody></table>

\


**Example**

```
// 성공 시
{
    "result" : {
        "code" : "Success",
        "message" : "Request has been completed successfully."
    },
    "subscription" : {
        "productId" : "다이아100_20170818000000",
        "productName" : "다이아100",
        "productAmount" : "2000",
        "productAmountMicros" : 2000000000,
        "priceCurrencyCode" : "KRW",
        "imagePath" : "https://xxx.png",
        "periodUnit" : "MONTH",
        "period" : 1,
        "purchaseToken" : "17070421461015116878",
        "status" : "SUBSCRIBING"
        "parentProductId" : "03904729375",
        "parentProductName" : "모상품명",
        "packageName" : "com.test.game",
        "startDate" : 1345578920000,
        "expiryDate" : 1345678920000,
        "startPaymentDate" : 1345578920000,
        "paymentAmount" : "1000",
        "paymentAmountMicros" : 1000000000,
        "nextPaymentAmount" : "1500",
        "nextPaymentAmountMicros" : 1500000000,
        "nextPaymentDate" : 1345678920000,
        "pauseAllow": "Y",
        "pauseStartDate" : 1625670000000,
        "pauseEndDate" : 1628840000000,
        "promotionAmount" : "1000",
        "promotionAmountMicros" : 1000000000,
        "promotionPeriod" : 1,
        "priceChanges" : [{
            "priceChangeSeq": 1,
            "priceChangeApplyStartDate": 1345678920000,
            "priceChangePreviousAmount" : "2000",
            "priceChangePreviousAmountMicros" : 2000000000,            
            "priceChangeAmount" : "2500",
            "priceChangeAmountMicros" : 2500000000,
            "priceChangeAgreement" : "N",
            "priceChangeAgreementDueDate": 1345678920000
        }]
    }
}
 
 
// 실패 시
{
    "error" : {
        "code" : "NoSuchData",
        "message" : "The requested data could not be found."
    }
}
```

## 결제

### 결제 요청 <a href="#id-2.-apiv7-5.1" id="id-2.-apiv7-5.1"></a>

원스토어 인 결제를 요청합니다. paymentUrl로 paymentParam을 전달합니다.&#x20;

성공 시, ONE store 표준 결제화면이 노출됩니다.

#### **\[ 호출 Spec. ]**

| Protocol     | HTTPS      | Method | POST       |
| ------------ | ---------- | ------ | ---------- |
| Content-Type | text/plain | Accept | text/plain |

<table data-header-hidden><thead><tr><th width="125.93359375"></th><th></th></tr></thead><tbody><tr><td>URL Path</td><td>paymentUrl</td></tr><tr><td>Description</td><td>웹 표준 결제화면 호출</td></tr></tbody></table>

**Parameter**

| Parameter Name | Data Size | Required | Description                         |
| -------------- | --------- | -------- | ----------------------------------- |
| paymentParam   | 500       | M        | 결제요청 데이터 (표준 결제화면 호출을 위한 parameter) |

#### **결제 요청 시 브라우저 크기**

원스토어 표준 결제화면에 최적화된 브라우저 크기는 width=400, height=580 입니다.&#x20;

브라우저 크기가 다른 경우 결제화면이 정상적으로 노출되지 않거나 동작하지 않을 수 있습니다. &#x20;

아래는 새 브라우저 창으로 원스토어 결제화면을 호출할 때의 예입니다.&#x20;

{% code overflow="wrap" %}
```
window.open('paymentUrl','pp01','width=400, height=580, left='+((window.screen.width / 2) - (400 / 2))+', top='+((window.screen.height / 2) - (580 / 2))+' status=no, menubar=no, toolbar=no, sizable=no');
```
{% endcode %}

### 결제 응답 <a href="#id-2.-apiv7-5.2" id="id-2.-apiv7-5.2"></a>

표준 결제화면의 결제결과를 개발사로 전달합니다.

#### **\[ 호출 Spec. ]**

| Signature Algorithm | SHA512 with RSA |
| ------------------- | --------------- |

| Protocol     | HTTP/HTTPS                                                                                | Method | POST                           |
| ------------ | ----------------------------------------------------------------------------------------- | ------ | ------------------------------ |
| Content-Type | <p>returnUrl : Application/x-www-form-urlencode<br><br>callbackUrl : Application/json</p> | Accept | callbackUrl : Application/json |

<table data-header-hidden><thead><tr><th width="125.15234375"></th><th></th></tr></thead><tbody><tr><td>URL Path</td><td>returnUrl, callbackUrl</td></tr><tr><td>Description</td><td><p>개발사가 제공한 returnUrl(Redirect Page)와 callbackUrl(REST API)을 통해 결제결과를 전달합니다. 두 방식 모두 Parameter Element는 동일하며 returnUrl은 form data submit 형태로, callbackUrl은 json 형태로 전달됩니다.</p><ul><li>단, callbackUrl 데이터는 실제 PG와의 연동 발생시 최종 결제결과에 대해서만 전달됩니다.</li></ul></td></tr></tbody></table>

**Parameter**

| Element Name          | Data Type | Data Size | Required | Description                                                                                                                                                                                                                                                       |
| --------------------- | --------- | --------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| responseCode          | String    | 20        | Y        | 응답코드 (하단 표 참조)                                                                                                                                                                                                                                                    |
| responseMessage       | String    | 200       | N        | 결제성공(Success)시 빈 값                                                                                                                                                                                                                                                |
| orderId               | String    | 20        | N        | 결제ID                                                                                                                                                                                                                                                              |
| purchaseId            | String    | 20        | N        | 구매ID                                                                                                                                                                                                                                                              |
| purchaseToken         | String    | 20        | N        | 구매토큰                                                                                                                                                                                                                                                              |
| purchaseTime          | Long      | 13        | N        | 구매시간(millisecond)                                                                                                                                                                                                                                                 |
| developerPayload      | String    | 200       | N        | 구매 건을 식별하기 위해 개발자에서 관리하는 식별자                                                                                                                                                                                                                                      |
| quantity              | Long      | 5         | N        | 복수구매 수량                                                                                                                                                                                                                                                           |
| purchaseSignature     | String    | 2000      | N        | <p>구매정보 검증을 위한 signature</p><ul><li><p>단건 결제시 </p><ul><li>(orderId+purchaseId+purchaseToken+purchaseTime+developerPayload)</li></ul></li><li><p>복수구매 결제시</p><ul><li>(orderId+purchaseId+purchaseToken+purchaseTime+developerPayload+quantity)</li></ul></li></ul> |
| <p>billingKey<br></p> | String    | 200       | N        | <p>S2S 자동결제 승인을 위한 billing 키<br></p>                                                                                                                                                                                                                              |

**Example(returnUrl)**

```
<form name="paymentResultForm" action="{returnUrl}" method="post">

<input type="hidden" name="responseCode" value="Success">
<input type="hidden" name="responseMessage" value="">
<input type="hidden" name="orderId" value="20200429OS01123456789">
<input type="hidden" name="purchaseId" value="20042912345678901234">
<input type="hidden" name="purchaseToken" value="20042912345678905678">
<input type="hidden" name="purchaseTime" value=5615474165165>
<input type="hidden" name="developerPayload" value="pd2020042912354987321">
<input type="hidden" name="quantity" value=3>
<input type="hidden" name="purchaseSignature" value="DB98B5CB92126B1D52E86FED4C6E4AC9E29ADAF356057DB98B5CB92126B1D5......">
<input type="hidden" name="billingKey" value="36FED4C6E4AC9E29ADAF356057DB98B5CB92126B1D52E87577....">

</form>

```

**Example(callbackUrl)**

```
{
  "responseCode" : "Success",
  "responseMessage" : "",
  "orderId" : "20200429OS01123456789",
  "purchaseId" : "20042912345678901234",
  "purchaseToken" : "20042912345678905678",
  "purchaseTime" : 5615474165165,
  "developerPayload" : "pd2020042912354987321",
  "quantity" : 3
  "purchaseSignature" : "DB98B5CB92126B1D52E86FED4C6E4AC9E29ADAF356057DB98B5CB92126B1D5......",
  "billingKey" : "36FED4C6E4AC9E29ADAF356057DB98B5CB92126B1D52E87577...."
}
```

#### **응답코드**

| Response Code      | Response Message     | Description                      |
| ------------------ | -------------------- | -------------------------------- |
| Success            | 빈값                   | 결제성공                             |
| Fail               | 설명참조                 | 각 PG사 및 내부 시스템 오류에 대한 원인을 전달합니다. |
| UserCancel         | 결제가 취소 되었습니다.        | <p><br></p>                      |
| PaymentTimeExpired | 결제시간이 초과 되었습니다.(10분) | <p><br></p>                      |

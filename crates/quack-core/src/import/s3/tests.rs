use aws_credential_types::Credentials;
use aws_credential_types::provider::SharedCredentialsProvider;

use super::*;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

fn object(url: &str) -> S3Object {
    S3Object::parse(&SourceUrl::from(url)).unwrap_or_else(|e| fail(&e.to_string()))
}

#[test]
fn an_s3_url_names_a_bucket_and_a_key() {
    let parsed = object("s3://sales-data/2026/q3/orders.parquet");
    assert_eq!(parsed.bucket, "sales-data");
    assert_eq!(parsed.key(), "2026/q3/orders.parquet");
    for bad in ["s3://bucket-only", "s3://bucket/", "s3:///key.csv"] {
        assert!(S3Object::parse(&SourceUrl::from(bad)).is_err(), "{bad}");
    }
}

#[test]
fn objects_are_addressed_virtual_hosted_unless_a_dot_or_an_endpoint_says_otherwise() {
    let aws = |fips| Endpoint::Aws {
        region: String::from("us-west-2"),
        fips,
    };
    let url = |o: &S3Object, e: &Endpoint| {
        o.url(e)
            .map_or_else(|e| fail(&e.to_string()), |u| u.to_string())
    };
    let plain = object("s3://sales-data/2026/orders file.csv");
    assert_eq!(
        url(&plain, &aws(false)),
        "https://sales-data.s3.us-west-2.amazonaws.com/2026/orders%20file.csv"
    );
    assert_eq!(
        url(&plain, &aws(true)),
        "https://sales-data.s3-fips.us-west-2.amazonaws.com/2026/orders%20file.csv"
    );
    assert_eq!(
        url(&object("s3://sales.example.com/a.csv"), &aws(false)),
        "https://s3.us-west-2.amazonaws.com/sales.example.com/a.csv"
    );
    assert_eq!(
        url(
            &plain,
            &Endpoint::Custom(String::from("http://127.0.0.1:9000/"))
        ),
        "http://127.0.0.1:9000/sales-data/2026/orders%20file.csv"
    );
}

/// The request carries S3's payload hash (of the empty body) and a
/// signature scoped to the region and the s3 service.
#[tokio::test]
async fn an_s3_get_is_signed_for_s3_with_the_payload_hash() {
    let credentials = SharedCredentialsProvider::new(Credentials::new(
        "AKIDEXAMPLE",
        "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        None,
        None,
        "test",
    ));
    let signed = Signer::s3(credentials, String::from("eu-central-1"))
        .signature(
            "GET",
            "https://sales-data.s3.eu-central-1.amazonaws.com/a.csv",
            &[],
            &[],
        )
        .await
        .unwrap_or_else(|e| fail(&e));
    let header = |name: &str| {
        signed
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
            .unwrap_or_default()
    };
    assert_eq!(
        header("x-amz-content-sha256"),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    let authorization = header("authorization");
    assert!(
        authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"),
        "{authorization}"
    );
    assert!(
        authorization.contains("/eu-central-1/s3/aws4_request"),
        "{authorization}"
    );
    assert!(
        authorization.contains("x-amz-content-sha256"),
        "{authorization}"
    );
}

//! A model that reads an image: `[ingestion].vision_model` describing an
//! uploaded picture at ingest, and a chat model marked `images = true` answering a question about one in a turn
//! (`view_image`). Each call is one tool-less request through the provider's
//! client, so it takes the provider's permit, keeps to the workspace's
//! allowed providers, and is retried like any other.

use std::time::Duration;

use base64::Engine as _;
use rig::message::{ImageMediaType, Message, UserContent};

use super::{ChatClient, PlainCall, Task};
use crate::config::{Config, ModelSettings};
use crate::error::Result;
use crate::ingestion::parser::ImageFormat;

/// What an ingested image becomes: its text first, as written, then what
/// it shows, so both keyword and vector search find it.
const DESCRIBE_PROMPT: &str = "You read an image so it can be searched and cited. First \
     transcribe every word visible in it, exactly as written, keeping its lines and any \
     table as Markdown. Then describe what the image shows: its kind (a photo, a chart, a \
     diagram, a scan, a screenshot), its subject, and any numbers, labels, axes, or \
     relationships it conveys. Write plain Markdown with no preamble. If the image holds no \
     text, say so in one line and describe it.";

/// What a question about an image in a turn is answered from.
const LOOK_PROMPT: &str = "You answer a question about one image for a data analysis \
     assistant. Answer from what the image shows, quoting its text exactly where it matters, \
     and say plainly when the image does not show what is asked.";

/// How long one image call may run.
const IMAGE_TIMEOUT: Duration = Duration::from_secs(300);

/// A model that takes images, set up for one purpose.
pub struct ImageReader(PlainCall);

impl ImageReader {
    /// The vision model `[ingestion].vision_model` names, at background
    /// effort, describing images; `None` when none is configured.
    ///
    /// # Errors
    ///
    /// Returns an error when the model names an unknown provider or its
    /// client cannot be built.
    pub async fn for_ingest(config: &Config) -> Result<Option<Self>> {
        let Some(model) = config.vision_model_ref()? else {
            return Ok(None);
        };
        let settings = config.model_settings(model);
        let chat = ChatClient::build(config, &model).await?.chat_model(
            model.model,
            settings.background_effort,
            settings.temperature,
        )?;
        Ok(Some(Self(PlainCall::new(
            chat,
            Task {
                preamble: DESCRIBE_PROMPT,
                timeout: IMAGE_TIMEOUT,
                label: "image reading",
            },
        ))))
    }

    /// The turn's chat model on `client`, when `images = true` marks it
    /// as one that reads images, for questions about one in a turn; `None`
    /// otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error when the model cannot be built on `client`.
    pub(crate) fn for_turn(
        client: &ChatClient,
        model: &str,
        settings: ModelSettings,
    ) -> Result<Option<Self>> {
        if settings.images != Some(true) {
            return Ok(None);
        }
        let chat = client.chat_model(model, settings.effort, settings.temperature)?;
        Ok(Some(Self(PlainCall::new(
            chat,
            Task {
                preamble: LOOK_PROMPT,
                timeout: IMAGE_TIMEOUT,
                label: "image question",
            },
        ))))
    }

    /// The model's answer about `image`, given `instruction` beside it.
    ///
    /// # Errors
    ///
    /// Returns the model's error, a cut-off answer, or a timeout.
    pub async fn read(
        &self,
        image: &[u8],
        format: ImageFormat,
        instruction: &str,
    ) -> Result<String> {
        let data = base64::engine::general_purpose::STANDARD.encode(image);
        let media = match format {
            ImageFormat::Png => ImageMediaType::PNG,
            ImageFormat::Jpeg => ImageMediaType::JPEG,
            ImageFormat::Webp => ImageMediaType::WEBP,
            ImageFormat::Gif => ImageMediaType::GIF,
        };
        let message = Message::User {
            content: vec![
                UserContent::image_base64(data, Some(media), None),
                UserContent::text(instruction),
            ],
        };
        Ok(self.0.text(message).await?.trim().to_owned())
    }
}

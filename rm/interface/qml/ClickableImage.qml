import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15


Item {
    id : root

    property alias image : image

    ColumnLayout {
        anchors.fill : parent

        Button {
            text : 'Select image'
        }

        Item {
            Layout.fillHeight : true
            Layout.fillWidth : true

            Image {
                id : image
                anchors.fill : parent

                antialiasing : true
                smooth : true
                fillMode : Image.PreserveAspectFit
                asynchronous : true

                // FIXME qt.gui.icc: fromIccProfile: failed minimal tag size sanity
                source : "../../resource/please stand by.jpg"
            }

            MouseArea {
                anchors.fill : parent

                onClicked : {
                    con.image_coord("mouseXY:(" + mouseX + "," + mouseY + ")" +
                    "|imageWH:(" + image.paintedWidth + "," + image.paintedHeight + ")" +
                    "|frameWH:(" + root.width + "," + root.height + ")")
                }
            }
        }
    }
}

package project.door.lock.pack;

import java.util.Random;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.telephony.SmsManager;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.EditText;

public class RegisterActivity extends Activity implements OnClickListener
{
	EditText txt_user,txt_pass,txt_number,txt_serial;
	Button btn_submit;
	
	public void onCreate(Bundle b)
	{
		super.onCreate(b);
		setContentView(R.layout.register);
		txt_user=(EditText)findViewById(R.id.register_txt_user);
		txt_pass=(EditText)findViewById(R.id.register_txt_pass);
		txt_number=(EditText)findViewById(R.id.register_txt_phone);
		txt_serial=(EditText)findViewById(R.id.register_txt_serial);
		btn_submit=(Button)findViewById(R.id.register_btn_submit);
		btn_submit.setOnClickListener(this);
		
	}

	public void onClick(View v) {
		// TODO Auto-generated method stub
		String num=txt_number.getText().toString();
		String user=txt_user.getText().toString();
		String pass=txt_pass.getText().toString();
		String serial=txt_serial.getText().toString();
		Random rand=new Random();
		int code=Math.abs(rand.nextInt());
		
		SmsManager smanager=SmsManager.getDefault();
		smanager.sendTextMessage(num,null,"Verification Code for the Door Lock System is:"+code,null,null);
		
		Intent it=new Intent(this,VerifyActivity.class);
		Bundle b1=new Bundle();
		b1.putInt("which",1);
		b1.putInt("code",code);
		b1.putString("number",num);
		b1.putString("user",user);
		b1.putString("pass",pass);
		b1.putString("serial",serial);
		it.putExtra("data",b1);
		startActivity(it);
		
	}

}

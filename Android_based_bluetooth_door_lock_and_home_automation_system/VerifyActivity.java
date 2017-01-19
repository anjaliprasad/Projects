package project.door.lock.pack;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.SharedPreferences.Editor;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.telephony.SmsManager;
import android.telephony.SmsMessage;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

public class VerifyActivity extends Activity implements OnClickListener
{
	EditText txt_verify;
	Button btn_submit;
	int which=1;
	int verification_code;
	String number,user,pass,serial;
	SharedPreferences prefs;
	
	public void onCreate(Bundle b)
	{
		super.onCreate(b);
		Bundle bdata=getIntent().getBundleExtra("data");
		which=bdata.getInt("which");
		if(which==1)
		{
			verification_code=bdata.getInt("code",1);
			number=bdata.getString("number");
			user=bdata.getString("user");
			pass=bdata.getString("pass");
			serial=bdata.getString("serial");
		}
		if(which==2)
		{
			SharedPreferences pref=PreferenceManager.getDefaultSharedPreferences(getBaseContext());
			String code=pref.getString("vcode","unknown");
			String number=pref.getString("number","9742061563");
			SmsManager smanager=SmsManager.getDefault();
			smanager.sendTextMessage(number,null,"Enter Verification Code:"+code+" to open the registration account",null,null);			
		}		
		setContentView(R.layout.verify);
		txt_verify=(EditText)findViewById(R.id.verify_txt_code);
		btn_submit=(Button)findViewById(R.id.verify_btn_submit);
		btn_submit.setOnClickListener(this);
	}

	public void onClick(View v) {
		// TODO Auto-generated method stub
		int code=Integer.parseInt(txt_verify.getText().toString());
		if(which==1)
		{
		
		if(code==verification_code)
		{
			
			
			prefs = PreferenceManager.getDefaultSharedPreferences(getBaseContext());
			Editor edit = prefs.edit();
			edit.putString("vcode",""+code);
			edit.putString("number",number);
			edit.putString("user",user);
			edit.putString("pass",pass);
			edit.putString("serial",serial);
			edit.commit();
			
			Intent it=new Intent(this,MainActivity.class);
			startActivity(it);
		}
		else
		{
			Toast.makeText(this,"Verification Failed",Toast.LENGTH_LONG).show();
		}
		}
		else
		{
			prefs = PreferenceManager.getDefaultSharedPreferences(getBaseContext());
			String code1=prefs.getString("vcode","Abc");
			if(code1.equals(code))
			{
				Intent it=new Intent(this,RegisterActivity.class);
				startActivity(it);
			}
			
		}
	}

}
